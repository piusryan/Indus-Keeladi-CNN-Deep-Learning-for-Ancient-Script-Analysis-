"""
Grid Decomposition Module for Indus-Keeladi CNN Project
Implements Figure 01 logic for decomposing complex signs into grid components
"""

import cv2
import numpy as np
from pathlib import Path


class GridDecomposer:
    """
    Decomposes Indus script signs into grid-based components
    Based on the 3x3 grid methodology from Figure 01
    """
    
    def __init__(self, grid_size=(3, 3)):
        """
        Initialize grid decomposer
        
        Args:
            grid_size: Tuple of (rows, cols) for the grid decomposition
        """
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
    
    def decompose_image(self, image):
        """
        Decompose image into grid cells
        
        Args:
            image: Input image array (grayscale)
            
        Returns:
            List of grid cell images
        """
        height, width = image.shape[:2]
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        grid_cells = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height if row < self.grid_rows - 1 else height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width if col < self.grid_cols - 1 else width
                
                cell = image[y_start:y_end, x_start:x_end]
                grid_cells.append(cell)
        
        return grid_cells
    
    def analyze_grid_presence(self, grid_cells, threshold=0.1):
        """
        Analyze which grid cells contain significant content
        
        Args:
            grid_cells: List of grid cell images
            threshold: Minimum pixel density to consider cell as "present"
            
        Returns:
            Binary grid presence matrix
        """
        presence_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        for idx, cell in enumerate(grid_cells):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Calculate pixel density
            if len(cell.shape) == 2:
                density = np.sum(cell > 0) / (cell.shape[0] * cell.shape[1])
            else:
                density = np.sum(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) > 0) / (cell.shape[0] * cell.shape[1])
            
            presence_matrix[row, col] = 1 if density > threshold else 0
        
        return presence_matrix
    
    def extract_features_from_grid(self, grid_cells):
        """
        Extract features from each grid cell
        
        Args:
            grid_cells: List of grid cell images
            
        Returns:
            Feature vector representing grid characteristics
        """
        features = []
        
        for cell in grid_cells:
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Basic features per cell
            cell_features = {
                'mean_intensity': np.mean(cell),
                'std_intensity': np.std(cell),
                'pixel_density': np.sum(cell > 0) / (cell.shape[0] * cell.shape[1]),
                'horizontal_symmetry': self._calculate_horizontal_symmetry(cell),
                'vertical_symmetry': self._calculate_vertical_symmetry(cell)
            }
            features.append(cell_features)
        
        return features
    
    def _calculate_horizontal_symmetry(self, cell):
        """Calculate horizontal symmetry score for a cell"""
        height = cell.shape[0]
        if height < 2:
            return 0.0
        
        top_half = cell[:height//2, :]
        bottom_half = cv2.flip(cell[height//2:, :], 0)
        
        # Pad if sizes don't match
        if top_half.shape != bottom_half.shape:
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_h, :]
            bottom_half = bottom_half[:min_h, :]
        
        return np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
    
    def _calculate_vertical_symmetry(self, cell):
        """Calculate vertical symmetry score for a cell"""
        width = cell.shape[1]
        if width < 2:
            return 0.0
        
        left_half = cell[:, :width//2]
        right_half = cv2.flip(cell[:, width//2:], 1)
        
        # Pad if sizes don't match
        if left_half.shape != right_half.shape:
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
        
        return np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
    
    def visualize_grid(self, image, presence_matrix=None, output_path=None):
        """
        Visualize grid decomposition on the original image
        
        Args:
            image: Original image
            presence_matrix: Optional presence matrix to highlight active cells
            output_path: Optional path to save visualization
            
        Returns:
            Image with grid overlay
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        height, width = image.shape[:2]
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Draw grid lines
        for i in range(1, self.grid_rows):
            y = i * cell_height
            cv2.line(vis_image, (0, y), (width, y), (0, 255, 0), 1)
        
        for i in range(1, self.grid_cols):
            x = i * cell_width
            cv2.line(vis_image, (x, 0), (x, height), (0, 255, 0), 1)
        
        # Highlight active cells if presence matrix provided
        if presence_matrix is not None:
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    if presence_matrix[row, col] == 1:
                        x_start = col * cell_width
                        y_start = row * cell_height
                        x_end = (col + 1) * cell_width if col < self.grid_cols - 1 else width
                        y_end = (row + 1) * cell_height if row < self.grid_rows - 1 else height
                        cv2.rectangle(vis_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Example usage
    decomposer = GridDecomposer(grid_size=(3, 3))
    print("Grid Decomposer initialized")
