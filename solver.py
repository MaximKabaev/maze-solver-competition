from importlib.resources import path
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
from collections import deque


class Maze():
    """
    A 2-D maze.
    
    Attributes
    ----------
    walls : numpy.ndarray
        2-D occupancy grid containing ones where there are walls and zeros 
        where there are open corridors.
    
    Author: Alan Hunter
    Date: 2024
    """
    
    def __init__(self,width,height,seed=None,animate=False):
        """
        Maze Constructor.
        
        Parameters
        ----------
        width : int
            Width of the maze in cells.
        height : int
            Height of the maze in cells.
        seed : int, optional
            Seed for the random number generator. The default is None for a 
            random seed.
        animate : bool, optional
            Animate the creation of the maze in a plot window. This is slow for 
            large mazes. The default is False.
            
        Returns
        -------
        None.
        """
        self.generate(width,height,seed,animate) 
    
    
    def generate(self,width_cells,height_cells,seed=None,animate=False):
        """
        Use Prim's algorithm to generate a random maze.
        
        Parameters
        ----------
        width_cells : int
            Width of the maze in cells.
        height_cells : TYPE
            Height of the maze in cells.
        seed : int, optional
            Seed for the random number generator. The default is None for a 
            random seed.
        animate : bool, optional
            Animate the creation of the maze in a plot window. This is slow for 
            large mazes. The default is False.
        
        Returns
        -------
        None.

        """
        
        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)
        
        # Array of cells
        #  0 : unexplored
        #  1 : explored
        # -1 : frontier
        cells = np.zeros([height_cells,width_cells])
        
        # Larger array for the walls and corridors
        # 0 : free
        # 1 : occupied
        width_walls = self.__convert_cell_to_walls(width_cells)
        height_walls = self.__convert_cell_to_walls(height_cells)
        
        walls = np.zeros([height_walls,width_walls])
        
        # Put walls between the cells
        walls[0::2,:] = 1
        walls[:,0::2] = 1
        
        # Possible directions of travel from a cell: North, South, East, West
        x_step = [0,0,1,-1]
        y_step = [1,-1,0,0]
        
        # Start from a random cell
        Ix_cells = int(np.floor( np.random.rand() * width_cells ))
        Iy_cells = int(np.floor( np.random.rand() * height_cells ))
        cells[Iy_cells,Ix_cells] = -1
        
        # Initialise plots
        if animate:
            
            # Interactive mode for animations
            plt.ion()
            fig = plt.figure()
            
            # Custom colourmaps
            cmap_cells = mpl.colors.ListedColormap(['darkorange','black','white'])
            cmap_walls = mpl.colors.ListedColormap(['white','black'])
            
            # Animation of cells
            plt.subplot(1,2,1)
            ax1 = plt.imshow(cells,origin='lower',cmap=cmap_cells)
            plt.title('Cells')
            plt.xticks([])
            plt.yticks([])
            plt.clim([-1,1])
            
            # Animation of walls
            plt.subplot(1,2,2)
            ax2 = plt.imshow(walls,origin='lower',cmap=cmap_walls)
            plt.title('Walls')
            plt.xticks([])
            plt.yticks([])
            plt.clim([0,1])
            
        # Loop until there are no frontier cells remaining
        while True:
            # Find frontier cells (i.e., cells that are unexplored but adjacent to an explored cell)
            [Iy_cells,Ix_cells] = np.where(cells == -1)
            
            if len(Ix_cells) == 0:
                # Algorithm is complete, so escape the loop
                break
            
            # Select a random frontier cell
            r = int( np.floor(np.random.rand()*len(Ix_cells)) )
            Ix_cells = Ix_cells[r]
            Iy_cells = Iy_cells[r]
            
            # Corresponding coordinates in the walls grid
            Ix_walls = self.__convert_cell_to_walls(Ix_cells)
            Iy_walls = self.__convert_cell_to_walls(Iy_cells)
            
            # Explore the frontier cell and connect it to another explored cell
            cells[Iy_cells,Ix_cells] = 1
            
            # Break a wall at random between this cell and 1 of 4 adjacent cells
            for p in np.random.permutation(4):
                x = x_step[p]
                y = y_step[p]
                
                # Check if wall is breakable (i.e., not the edges)
                if (Ix_cells+x >= 0) & (Ix_cells+x <= width_cells-1) & (Iy_cells+y >= 0) & (Iy_cells+y <= height_cells-1):
                    
                    # Check if adjacent cell has been explored
                    if cells[Iy_cells+y,Ix_cells+x] == 1:
                        
                        # Break the wall to this cell
                        walls[Iy_walls+y,Ix_walls+x] = 0
                        break
            
            # Label new frontier cells
            for p in range(4):
                x = x_step[p];
                y = y_step[p];
                
                if (Ix_cells+x >= 0) & (Ix_cells+x <= width_cells-1) & (Iy_cells+y >= 0 ) & (Iy_cells+y <= height_cells-1):
                    if cells[Iy_cells+y,Ix_cells+x] != 1:
                        cells[Iy_cells+y,Ix_cells+x] = -1
            
            # Plot progress
            if animate:
                ax1.set_data(cells)
                ax2.set_data(walls)
                plt.pause(0.01)
                
        if animate:
            plt.close(fig)
        
        # Finally, break external walls at random locations on opposite sides
        if np.random.rand() > width_cells/(width_cells+height_cells):
            # North / South sides
            r1 = self.__convert_cell_to_walls(int(np.floor(np.random.rand()*width_cells)))
            r2 = self.__convert_cell_to_walls(int(np.floor(np.random.rand()*width_cells)))
            walls[0,r1] = 0
            walls[-1,r2] = 0
        else:
            # East / West sides
            r1 = self.__convert_cell_to_walls(int(np.floor(np.random.rand()*height_cells)))
            r2 = self.__convert_cell_to_walls(int(np.floor(np.random.rand()*height_cells)))
            walls[r1,0] = 0
            walls[r2,-1] = 0
            
        # Maze generation complete
        self.walls = walls
        
    
    def __convert_cell_to_walls(self,x):
        """
        Convert between coordinates of cell grid and occupancy grid.
        
        Parameters
        ----------
        x : int
            Cell coordinate.
        
        Returns
        -------
        y : int
            Occupancy coordinate.
        """
        y = 2*x+1
        return y
    
    
    def __str__(self):
        """
        Text-based representation of the maze.
        """
        s = ''
        for m in range(self.walls.shape[0]-1,-1,-1):
            for n in range(self.walls.shape[1]):
                if self.walls[m,n] == 1:
                    s = s + '\u2588\u2588'
                else:
                    s = s + '\u2591\u2591'
            s = s + '\n'
        return s
    
    
    def plot(self,show_indices=False,path=None):
        """
        Visual representation of the maze.
        
        Parameters
        ----------
        show_indices : bool, optional
            Show the indices of the maze coordinates. The default is False.

        Returns
        -------
        None.
        """
        
        fig = plt.figure()
        
        ax = plt.imshow(self.walls,origin='lower',cmap=mpl.colors.ListedColormap(['white','black']))
        plt.clim([0,1])

        if path:
            path_coords = np.array(path)
            plt.plot(path_coords[:, 1], path_coords[:, 0], 'r-', linewidth=2)  # red line for path
            plt.plot(path_coords[0, 1], path_coords[0, 0], 'go')  # green dot for start
            plt.plot(path_coords[-1, 1], path_coords[-1, 0], 'ro')  # red dot for end
    
        if show_indices:
            ny, nx = self.walls.shape
            plt.xticks(np.linspace(0,nx-1,nx))
            plt.yticks(np.linspace(0,ny-1,ny))
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
        
        plt.show(block=True)
    
    def get_node_neighbours(self, row, col):
        """
        Get the neighbouring nodes of a given node in the maze.

        Parameters
        ----------
        row : int
            Row index of the node.
        col : int
            Column index of the node.

        Returns
        -------
        neighbours : list of tuples
            List of (row, col) tuples of neighbouring nodes.
        """

        neighbours = []
        height, width = self.walls.shape

        for r in range (-1,2,2):
            try:
                if self.walls[row+r, col] == 0 and row+r < height:
                    neighbours.append((row+r, col))
            except IndexError:
                continue
        for c in range (-1,2,2):
            try:
                if self.walls[row, col+c] == 0 and col+c < width:
                    neighbours.append((row, col+c))
            except IndexError:
                continue

        return neighbours
    
    def get_start_end_nodes(self):
        """
        Find the start and end nodes of the maze (two nodes at the edge of the maze that are not walls).

        Returns
        -------
        start : tuple
            (row, col) of start node.
            end : tuple
            (row, col) of end node.
        """
        edge_nodes = []
        height, width = self.walls.shape
        for w in range (width):
            if self.walls[0, w] == 0:
                edge_nodes.append((0, w))
            if self.walls[height-1, w] == 0:
                edge_nodes.append((height-1, w))
        for h in range (height):
            if self.walls[h, 0] == 0:
                edge_nodes.append((h, 0))
            if self.walls[h, width-1] == 0:
                edge_nodes.append((h, width-1))
        return edge_nodes[0], edge_nodes[1]
    
    def solve(self):
        """
        Solve the maze using Depth-First Search (DFS) algorithm.

        Returns
        -------
        path : list of tuples
            List of (row, col) tuples representing the solution path from start to end.
        """

        start, end = self.get_start_end_nodes()
        frontier = [start]
        visited = {start: None}
        while frontier:
            current = frontier.pop()
            if current == end:
                break
            for neighbour in self.get_node_neighbours(current[0], current[1]):
                if neighbour not in visited:
                    frontier.append(neighbour)
                    visited[neighbour] = current

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = visited.get(current)
    
        return list(reversed(path))


if __name__ == '__main__':
    
    # Create a test maze
    maze = Maze(150,100)
    
    # View the maze
    print(maze)
    
    # Solve the maze
    start_time = time.time()
    solution = maze.solve()
    end_time = time.time()

    solve_time = end_time - start_time

    print(f"Maze solved in {solve_time:.4f} seconds")
    print(f"Solution length: {len(solution)} steps")
    
    # View the maze with solution
    maze.plot(path=solution)
