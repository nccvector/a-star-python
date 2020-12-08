import numpy as np
from cv2 import cv2
import math
import time
import copy

def valid(grid):
    color_display = np.ones((grid.shape[0],grid.shape[1],3), dtype=np.uint8) * 255
    color_display[grid == 0] = 0

    # Displaying initial grid
    cv2.imshow('grid', cv2.resize(color_display, (600,500), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(10)

    # Generating inf cost array
    cost_grid = np.ones(grid.shape, dtype=np.int) * np.inf

    # Generating visited array
    opened = np.zeros(grid.shape, dtype=np.int)
    closed = np.zeros(grid.shape, dtype=np.int)

    # Generating parent grid
    parent_grid = np.ones((grid.shape[0], grid.shape[1], 2), dtype=np.int) * -1

    start_position = [0,0]
    cost_grid[start_position[0], start_position[1]] = 0
    end_position = [grid.shape[0]-1, grid.shape[1]-1]
    # end_position = [60,20]

    current_position = copy.deepcopy(start_position)

    # Adding starting position in open queue
    queue = [current_position]
    while not current_position == end_position:

        # Getting the node with least f_cost
        # Updated queue now need to select current node from queue
        min_cost = np.inf
        for position in queue:
            if cost_grid[position[0],position[1]] < min_cost:
                min_cost = cost_grid[position[0],position[1]]
                current_position = copy.deepcopy(position)

        # Removing current node from open queue
        try:
            opened[current_position[0],current_position[1]] = 0
            queue.remove(current_position)
        except:
            # pass
            return None

        # Marking current node as closed
        # Marking current node as visited
        closed[current_position[0],current_position[1]] = 1

        # Break the search if reached goal
        if current_position == end_position:
            break

        # Updating queue
        for y in range(-1,2):
            pos_y = y + current_position[1]
            for x in range(-1,2):
                pos_x = x + current_position[0]
                if not (x+y) == 0 and pos_y > -1 and pos_y < grid.shape[0] and pos_x > -1 and pos_x < grid.shape[1]\
                    and grid[pos_x,pos_y] == 1 and closed[pos_x,pos_y] == 0\
                    and (abs(x) + abs(y)) == 1: # Remove this last condition for possibility of diagonal paths

                    # Use this cost for standard Dijkstra (No heuristic)
                    # new_cost = cost_grid[current_position[0],current_position[1]] + \
                    # ((x)**2 + (y)**2)**0.5

                    # Use this cost for standard A* (It has distance to goal as a heuristic)
                    new_cost = cost_grid[current_position[0],current_position[1]] + \
                    ((x)**2 + (y)**2)**0.5 + \
                    ((end_position[0]-pos_x)**2 + (end_position[1]-pos_y)**2) ** 2
                    
                    if new_cost < cost_grid[pos_x,pos_y] or opened[pos_x,pos_y] == 0:
                        cost_grid[pos_x,pos_y] = new_cost
                        parent_grid[pos_x,pos_y,0] = current_position[0]
                        parent_grid[pos_x,pos_y,1] = current_position[1]

                        if opened[pos_x, pos_y] == 0:
                            opened[pos_x, pos_y] = 1
                            queue.append([pos_x,pos_y])

        color_display[closed == 1] = (255,0,0)
        color_display[current_position[0],current_position[1],:] = (0,0,255)
        cv2.imshow('grid', cv2.resize(color_display, (600,500), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(10)

    path = [current_position]
    while not current_position == start_position:
        parent_y = parent_grid[current_position[0],current_position[1],0]
        parent_x = parent_grid[current_position[0],current_position[1],1]
        current_position = [parent_y, parent_x]
        path.append(current_position)

    for coord in path:
        color_display[coord[0],coord[1],:] = (0,255,0)
    
    cv2.imshow('grid', cv2.resize(color_display, (600,500), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(10)
    time.sleep(3)

    return True


def generate_symetric_world(size=20):

    # Size must be a multiple of 2
    world = np.zeros((size, size), dtype=np.float)
    noise_array = np.zeros((int(size/2), int(size/2)), dtype=np.float)

    for y in range(noise_array.shape[0]):
        for x in range(noise_array.shape[1]):
            noise_array[y,x] = np.random.rand()

    # Clearing spawn positions
    noise_array[0,0] = 1

    # Assigning 4 parts of the world the rotated version of noise

    world[:int(world.shape[0]/2),:int(world.shape[1]/2)] = noise_array
    # Rotating noise array 90
    noise_array = np.rot90(noise_array)

    world[int(world.shape[0]/2):,:int(world.shape[1]/2)] = noise_array

    # Rotating noise array 90
    noise_array = np.rot90(noise_array)

    world[int(world.shape[0]/2):,int(world.shape[1]/2):] = noise_array

    # Rotating noise array 90
    noise_array = np.rot90(noise_array)

    world[:int(world.shape[0]/2),int(world.shape[1]/2):] = noise_array

    thresh_val = 0.35
    # Threshold the world until there are possible paths between two spawn positions
    _, thresh_world = cv2.threshold(world.copy(), thresh_val, thresh_val, cv2.THRESH_BINARY)
    thresh_world[thresh_world>0] = 1

    valid(thresh_world)

    return thresh_world


if __name__ == "__main__":

    while True:
        generate_symetric_world(size=64)