from collections import deque

def water_jug_bfs(capacity_x=4, capacity_y=3, target=2):
    start = (0, 0)

    queue = deque([(start, [])])

    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        if x == target:
            return path + [(x, y)]

        next_states = []

        next_states.append((capacity_x, y))

        next_states.append((x, capacity_y))

        next_states.append((0, y))


        next_states.append((x, 0))


        pour_xy = min(x, capacity_y - y)
        next_states.append((x - pour_xy, y + pour_xy))


        pour_yx = min(y, capacity_x - x)
        next_states.append((x + pour_yx, y - pour_yx))

        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [(x, y)]))

    return None


solution_path = water_jug_bfs()

if solution_path:
    print("Solution path:")
    for step in solution_path:
        print(step)
else:
    print("No solution found.")