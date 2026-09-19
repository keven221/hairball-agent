"""Kanban Goal-root view (``is_goal=1``) — not the session ``/goal`` slash command.

Thin CLI + helpers over Kanban tasks marked as user-intent Goal roots.
Distinct from:
- session ``/goal`` (Ralph loop in SessionDB via ``hairball_cli.goals``)
- worker ``goal_mode`` / ``kanban create --goal`` (Ralph on a card)

Restored from ``hairball_cli/__pycache__/goal_kanban.cpython-312.pyc`` after
the ``.py`` source was lost (Goals UI was 500ing on import).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from typing import Any, Optional

from hairball_cli import kanban_db as kb

_CRITERIA_RE = re.compile(
    r"\*\*Acceptance criteria\*\*\s*[—\-–:]?\s*(.*?)(?=\n\s*\*\*[^*]+\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_TERMINAL_DONE = frozenset({"done", "archived"})


def ensure_is_goal_column(conn) -> None:
    """Additive Hairball column — kept out of legacy-parity ``kanban_db`` defs."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "is_goal" not in cols:
        conn.execute(
            "ALTER TABLE tasks ADD COLUMN is_goal INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_is_goal ON tasks(is_goal)"
        )


def _task_from_goal_row(row) -> Any:
    task = kb.Task.from_row(row)
    # legacy ``from_row`` does not bind ``is_goal``; force from the SQL row.
    try:
        task.is_goal = bool(row["is_goal"]) if "is_goal" in row.keys() and row["is_goal"] else False
    except Exception:
        task.is_goal = False
    return task


def list_goal_roots(
    conn,
    *,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    include_archived: bool = False,
) -> list[Any]:
    """List Kanban tasks with ``is_goal=1`` (Goals UI / ``hairball goal``)."""
    ensure_is_goal_column(conn)
    query = "SELECT * FROM tasks WHERE is_goal = 1"
    params: list[Any] = []
    if status is not None:
        if status not in kb.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(kb.VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    query += " ORDER BY priority DESC, created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [_task_from_goal_row(r) for r in rows]


def mark_is_goal(conn, task_id: str, value: bool = True) -> None:
    ensure_is_goal_column(conn)
    conn.execute(
        "UPDATE tasks SET is_goal = ? WHERE id = ?",
        (1 if value else 0, task_id),
    )


def create_goal_root(
    conn,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: str = "user",
    tenant: Optional[str] = None,
    priority: int = 0,
    triage: bool = False,
) -> str:
    """Create a task then mark it ``is_goal=1`` (legacy ``create_task`` has no flag)."""
    ensure_is_goal_column(conn)
    task_id = kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        created_by=created_by,
        tenant=tenant,
        priority=priority,
        triage=triage,
        goal_mode=False,
    )
    mark_is_goal(conn, task_id, True)
    return task_id


def get_task_with_goal_flag(conn, task_id: str) -> Any:
    ensure_is_goal_column(conn)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return _task_from_goal_row(row)


def extract_acceptance_criteria(body: Optional[str]) -> str:
    """Parse the ``**Acceptance criteria**`` section from a specify-style body."""
    text = str(body or "").strip()
    if not text:
        return ""
    match = _CRITERIA_RE.search(text)
    return (match.group(1) or "").strip() if match else ""


def strip_acceptance_criteria(body: Optional[str]) -> Optional[str]:
    """Return body with the ``**Acceptance criteria**`` section removed.

    Used by UI so the description panel does not duplicate the criteria panel.
    """
    text = str(body or "").strip()
    if not text:
        return None
    match = _CRITERIA_RE.search(text)
    if not match:
        return text or None
    cleaned = (text[: match.start()] + text[match.end() :]).strip()
    return cleaned or None


def build_goal_tree(
    conn,
    root_id: str,
    max_nodes: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a Goal progress tree rooted at ``root_id``.

    Kanban ``task_links`` are **dependency** edges (parent must finish
    before child runs). ``kanban decompose`` therefore links the Goal
    root as a *child* of its work items so the root waits on them.

    Tree walk preference:
    1. **waits_on** — walk ``parent_ids`` (decompose-native; Goal waits
       on work packages).
    2. **children** — walk ``child_ids`` only when the root has no
       parents (manual fan-out via ``create --parent <goal>``).

    Returns ``(nodes, progress)`` where each node is
    ``{id, title, status, depth, is_goal, orientation}`` and progress is
    ``{done, total, blocked, running, orientation, work_done, work_total}``.
    ``done/total`` cover the whole tree (root included); ``work_*`` exclude
    the Goal root so UI progress reflects execution packages only.
    """
    root = get_task_with_goal_flag(conn, root_id)
    if root is None:
        raise ValueError(f"unknown task: {root_id}")

    direct_parents = kb.parent_ids(conn, root_id)
    direct_children = kb.child_ids(conn, root_id)
    if direct_parents:
        orientation = "waits_on"

        def neighbors(tid: str) -> list[str]:
            return kb.parent_ids(conn, tid)

    else:
        orientation = "children"

        def neighbors(tid: str) -> list[str]:
            return kb.child_ids(conn, tid)

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(root_id, 0)]
    while queue:
        tid, depth = queue.pop(0)
        if tid in seen:
            continue
        seen.add(tid)
        task = get_task_with_goal_flag(conn, tid)
        if task is None:
            continue
        nodes.append(
            {
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "depth": depth,
                "is_goal": bool(getattr(task, "is_goal", False)),
                "assignee": task.assignee,
                "orientation": orientation,
            }
        )
        if len(nodes) >= max_nodes:
            break
        for nid in neighbors(tid):
            if nid not in seen:
                queue.append((nid, depth + 1))

    # Decompose polarity edge case: root has parents listed empty after
    # incomplete link state, but children exist — fall back to children walk.
    if orientation == "waits_on" and len(nodes) == 1 and direct_children:
        orientation = "children"
        nodes = []
        seen = set()
        queue = [(root_id, 0)]
        while queue:
            tid, depth = queue.pop(0)
            if tid in seen:
                continue
            seen.add(tid)
            task = get_task_with_goal_flag(conn, tid)
            if task is None:
                continue
            nodes.append(
                {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status,
                    "depth": depth,
                    "is_goal": bool(getattr(task, "is_goal", False)),
                    "assignee": task.assignee,
                    "orientation": orientation,
                }
            )
            if len(nodes) >= max_nodes:
                break
            for nid in kb.child_ids(conn, tid):
                if nid not in seen:
                    queue.append((nid, depth + 1))

    total = len(nodes)
    done = sum(1 for n in nodes if str(n.get("status") or "") in _TERMINAL_DONE)
    blocked = sum(1 for n in nodes if str(n.get("status") or "") == "blocked")
    running = sum(1 for n in nodes if str(n.get("status") or "") == "running")
    work_nodes = [n for n in nodes if n.get("id") != root_id]
    work_total = len(work_nodes)
    work_done = sum(1 for n in work_nodes if str(n.get("status") or "") in _TERMINAL_DONE)
    progress = {
        "done": done,
        "total": total,
        "blocked": blocked,
        "running": running,
        "orientation": orientation,
        "work_done": work_done,
        "work_total": work_total,
    }
    return nodes, progress


def _task_brief(t: Any) -> dict[str, Any]:
    return {
        "id": t.id,
        "title": t.title,
        "body": t.body,
        "status": t.status,
        "assignee": t.assignee,
        "priority": t.priority,
        "tenant": t.tenant,
        "is_goal": bool(getattr(t, "is_goal", False)),
        "goal_mode": bool(getattr(t, "goal_mode", False)),
        "created_at": t.created_at,
        "acceptance_criteria": extract_acceptance_criteria(t.body),
    }


def _env_board() -> Optional[str]:
    raw = str(os.environ.get("HAIRBALL_KANBAN_BOARD") or "").strip()
    return raw or None


def build_parser(parent_subparsers) -> argparse.ArgumentParser:
    """Attach the ``goal`` subcommand tree. Returns the top parser."""
    goal = parent_subparsers.add_parser(
        "goal",
        help="User-intent Goal view over Kanban (not the /goal slash command)",
        description=(
            "Thin Goal view over Kanban tasks marked is_goal=1. This is NOT the "
            "interactive slash command `/goal` (session Ralph loop) and NOT "
            "`kanban create --goal` (worker goal_mode). Goals are durable Kanban "
            "roots; children use task_links."
        ),
    )
    goal.add_argument(
        "--board",
        metavar="<slug>",
        help=(
            "Board slug to operate on. Defaults to the current board (same "
            "resolution as `hairball kanban --board`)."
        ),
    )
    sub = goal.add_subparsers(dest="goal_action")

    create = sub.add_parser("create", help="Create a Goal root task on the Kanban board")
    create.add_argument("title", help="Goal title (user-facing intent)")
    create.add_argument("--body", default=None, help="Optional body / spec")
    create.add_argument("--assignee", default=None, help="Profile assignee")
    create.add_argument("--tenant", default=None, help="Tenant namespace")
    create.add_argument("--priority", type=int, default=0)
    create.add_argument(
        "--triage",
        action="store_true",
        help="Park in triage for specify/decompose (recommended for new goals)",
    )
    create.add_argument(
        "--created-by",
        default="user",
        help="Author recorded on the task (default: user)",
    )
    create.add_argument("--json", action="store_true", help="Emit JSON")

    listing = sub.add_parser("list", aliases=["ls"], help="List Goal root tasks (is_goal=1)")
    listing.add_argument(
        "--status",
        choices=sorted(kb.VALID_STATUSES),
        default=None,
        help="Filter by status",
    )
    listing.add_argument(
        "--all",
        dest="include_archived",
        action="store_true",
        help="Include archived goals",
    )
    listing.add_argument("--json", action="store_true", help="Emit JSON")
    listing.add_argument("--tenant", default=None)

    show = sub.add_parser("show", help="Show a Goal and optional dependency tree")
    show.add_argument("task_id", help="Goal (or any) task id")
    show.add_argument("--tree", action="store_true", help="Include dependency tree")
    show.add_argument("--json", action="store_true", help="Emit JSON")
    show.add_argument(
        "--criteria",
        action="store_true",
        help="Print acceptance criteria section",
    )

    goal.set_defaults(_goal_parser=goal)
    return goal


def _cmd_create(args) -> int:
    with kb.connect_closing() as conn:
        task_id = create_goal_root(
            conn,
            title=args.title,
            body=args.body,
            assignee=args.assignee,
            created_by=args.created_by,
            tenant=args.tenant,
            priority=args.priority,
            triage=bool(getattr(args, "triage", False)),
        )
        task = get_task_with_goal_flag(conn, task_id)
    assert task is not None
    if getattr(args, "json", False):
        print(json.dumps(_task_brief(task), indent=2, ensure_ascii=False))
        return 0
    print(
        f"Created goal {task_id}  (status={task.status}, is_goal=1, "
        f"assignee={task.assignee or '-'})"
    )
    print(
        "Note: this is a Kanban Goal root — not the /goal slash command, "
        "and not worker --goal mode.",
        file=sys.stderr,
    )
    return 0


def _cmd_list(args) -> int:
    with kb.connect_closing() as conn:
        tasks = list_goal_roots(
            conn,
            status=getattr(args, "status", None),
            tenant=getattr(args, "tenant", None),
            include_archived=bool(getattr(args, "include_archived", False)),
        )
    if getattr(args, "json", False):
        print(
            json.dumps([_task_brief(t) for t in tasks], indent=2, ensure_ascii=False)
        )
        return 0
    if not tasks:
        print("(no goals)")
        return 0
    for t in tasks:
        assignee = t.assignee or "-"
        print(f"  [{t.status:9s}]  {t.id}  {t.title}  (assignee={assignee})")
    return 0


def _cmd_show(args) -> int:
    with kb.connect_closing() as conn:
        task = get_task_with_goal_flag(conn, args.task_id)
        if task is None:
            print(f"goal: unknown task {args.task_id}", file=sys.stderr)
            return 1
        nodes = None
        progress = None
        if getattr(args, "tree", False):
            nodes, progress = build_goal_tree(conn, args.task_id)
    if getattr(args, "json", False):
        payload = _task_brief(task)
        if nodes is not None:
            payload["tree"] = nodes
            payload["progress"] = progress
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"{task.id}  [{task.status}]  {task.title}")
    if getattr(args, "criteria", False):
        criteria = extract_acceptance_criteria(task.body)
        print("Acceptance criteria:")
        print(criteria or "(none)")
    if nodes is not None and progress is not None:
        print(
            f"Progress: {progress.get('work_done', 0)}/{progress.get('work_total', 0)} "
            f"work ({progress.get('orientation')})"
        )
        for n in nodes:
            indent = "  " * int(n.get("depth") or 0)
            flag = "★" if n.get("is_goal") else "·"
            print(f"{indent}{flag} [{n.get('status')}] {n.get('id')} {n.get('title')}")
    return 0


def goal_command(args) -> int:
    """Entry point from ``hairball goal …`` argparse dispatch."""
    action = getattr(args, "goal_action", None)
    if not action:
        parser = getattr(args, "_goal_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hairball goal <create|list|show> [options]\n"
                "Note: this is NOT the interactive /goal slash command.\n"
                "Run 'hairball goal --help' for details.",
                file=sys.stderr,
            )
        return 2

    board = getattr(args, "board", None)
    # Board env pinning matches kanban CLI; optional when unset.
    with contextlib.nullcontext():
        if board:
            try:
                slug = kb._normalize_board_slug(board)  # type: ignore[attr-defined]
            except Exception:
                slug = str(board).strip()
            if not slug:
                print("goal: --board requires a slug", file=sys.stderr)
                return 2
            if hasattr(kb, "board_exists") and not kb.board_exists(slug):
                print(
                    f"goal: board {slug} does not exist. Create it with "
                    f"`hairball kanban boards create {slug}`.",
                    file=sys.stderr,
                )
                return 2
            os.environ["HAIRBALL_KANBAN_BOARD"] = slug
        elif _env_board():
            pass

        if action == "create":
            return _cmd_create(args)
        if action in {"list", "ls"}:
            return _cmd_list(args)
        if action == "show":
            return _cmd_show(args)
        print(f"goal: unknown action {action}", file=sys.stderr)
        return 2
