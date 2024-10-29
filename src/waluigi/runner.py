from waluigi import logger
from waluigi.task import *
from waluigi.graph import Graph, Left, Right
from waluigi.errors import *
from waluigi.bundle import *

import copy
import asyncio
from contextlib import asynccontextmanager
from dataclasses import field



def add_task(graph: Graph, task: Task):
    """
    add_task(graph, task) adds a task and all of its
    dependencies to the graph, recursively.
    Tasks that are already done are added as dependencies,
    to tasks that depend on them, but their dependencies 
    are not added.
    """
    def inner(parent):
        for child in parent._requires():
            if not (child.done() or graph.has(child)):
                inner(child)
            graph.add(child, parent)

    if not (task.done() or graph.has(task)):
        inner(task)
        graph.add(task)

def mk_dag(*tasks):
    """
    tasks: [Task]

    Construct the graph for the given tasks.
    Returns a list in topological order of type
    [(Task, TaskInfo)], where TaskInfo contains
    dependents and dependencies of the task in 
    question.
    """
    graph = Graph()
    for task in tasks:
        add_task(graph, task)
    leftmost, *sorted, rightmost = graph.toposort()
    assert rightmost == graph.rightmost.val
    assert leftmost == graph.leftmost.val
    return [(val, graph.get(val)) for val in sorted]

async def run_dag(task_edges, **kwargs):
    """
    task_edges: [(Task, TaskInfo)]
    kwargs: keyword arguments passed to the run_async methods of the tasks.

    Schedules all tasks in the dag, ensuring that downstreams tasks await
    upstreams tasks, and ensures that tasks with cleanup runs after dependent
    tasks are finished.

    Prints out task results on finish.
    """
    runs = {}
    cleanups = {}
    done = 0
    for (task, edges) in task_edges:
        if task.done():
            runs[task] = asyncio.create_task(task.noop())
            done += 1
        else:
            deps = [runs[l] for l in edges.left]
            runs[task] = asyncio.create_task(task._run_after(*deps, **kwargs))

    for (task, edges) in task_edges:
        if isinstance(task, TaskWithCleanup):
            deps = [runs[r] for r in edges.right]
            if not task.done():
                deps += [runs[task]]
            cleanups[task] = asyncio.create_task(task._cleanup_after(*deps, **kwargs))
    
    logger.info('Tasks scheduled, starting run')
    results = await asyncio.gather(*runs.values(), *cleanups.values(), return_exceptions=True)
    run_results, clean_results = results[:len(runs.values())], results[len(runs.values()):]
    log_results(done, run_results, clean_results)

def log_results(done, run_results, clean_results):
    """
    friendly logging of run results.
    """
    runs = 0
    run_fails = []
    run_depfails = 0
    cleans = 0
    clean_fails = []
    clean_depfails = 0

    for result in run_results:
        match result:
            case Task():
                runs += 1
            case FailedRun() as fail:
                run_fails.append(fail)
            case FailedDependency():
                run_depfails += 1
            case _:
                assert False, 'Pattern matching error'

    for result in clean_results:
        match result:
            case Task():
                cleans += 1
            case FailedRun() as fail:
                clean_fails.append(fail)
            case FailedDependency():
                clean_depfails += 1
            case _:
                assert False, 'Pattern matching error'

    all_ok = not (run_fails or clean_fails)

    if run_fails:
        logger.warning('======== RUN ERRORS ==========')
        for fail in run_fails:
            try:
                raise fail
            except FailedRun as e:
                logger.exception('run failure')

    if clean_fails:
        logger.warning('======= CLEANUP ERRORS =======')
        for fail in clean_fails:
            try:
                raise fail
            except FailedCleanup as e:
                logger.exception('cleanup failure')
        
    logger.info('======== RUN STATUS ==========')

    if not all_ok:
        logger.warning('~~~~~~~~~ bad news ~~~~~~~~~~~')
        logger.warning(f'Run failures          : {len(run_fails)}')
        logger.warning(f'  dependency failures : {run_depfails}')
        logger.warning(f'Clean failures        : {len(clean_fails)}')
        logger.warning(f'  dependency failures : {clean_depfails}')
    
    logger.info('~~~~~~~~~ good news ~~~~~~~~~~')
    logger.info(f'Already existing : {done}')
    logger.info(f'Run successes    : {runs - done} / {len(run_results) - done}')
    logger.info(f'Clean suncceses  : {cleans} / {len(clean_results)}')

    if all_ok:
        logger.info('All tasks successfull! :)')
    else:
        logger.warning('There were failed tasks! :(')
