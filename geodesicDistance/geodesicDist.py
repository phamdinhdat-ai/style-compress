import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import trimesh
from sklearn.neighbors import NearestNeighbors

class GeodesicDistanceCalculator:
    """
    A comprehensive class for calculating geodesic distances on 3D surfaces
    using various methods including analytical solutions, mesh-based algorithms,
    and numerical methods.
    """
    
    def __init__(self):
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.surface_type = None
        
    def create_sphere(self, radius=1.0, resolution=50):
        """Create a sphere mesh"""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Create triangular faces
        faces = []
        for i in range(resolution-1):
            for j in range(resolution-1):
                v1 = i * resolution + j
                v2 = i * resolution + j + 1
                v3 = (i + 1) * resolution + j
                v4 = (i + 1) * resolution + j + 1
                
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.surface_type = 'sphere'
        self.radius = radius
        return self
    
    def create_cylinder(self, radius=1.0, height=2.0, resolution=50):
        """Create a cylinder mesh"""
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(-height/2, height/2, resolution)
        theta, z = np.meshgrid(theta, z)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Create faces for cylinder surface
        faces = []
        for i in range(resolution-1):
            for j in range(resolution-1):
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1) % resolution
                v3 = ((i + 1) % resolution) * resolution + j
                v4 = ((i + 1) % resolution) * resolution + (j + 1) % resolution
                
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.surface_type = 'cylinder'
        self.radius = radius
        self.height = height
        return self
    
    def create_torus(self, major_radius=2.0, minor_radius=0.5, resolution=50):
        """Create a torus mesh"""
        u = np.linspace(0, 2*np.pi, resolution)
        v = np.linspace(0, 2*np.pi, resolution)
        u, v = np.meshgrid(u, v)
        
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)
        
        vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Create faces
        faces = []
        for i in range(resolution-1):
            for j in range(resolution-1):
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1) % resolution
                v3 = ((i + 1) % resolution) * resolution + j
                v4 = ((i + 1) % resolution) * resolution + (j + 1) % resolution
                
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        self.vertices = vertices
        self.faces = np.array(faces)
        self.surface_type = 'torus'
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        return self
    
    def euclidean_distance(self, point1, point2):
        """Calculate straight-line Euclidean distance"""
        return np.linalg.norm(point1 - point2)
    
    def analytical_geodesic_distance(self, point1, point2):
        """
        Calculate geodesic distance using analytical formulas
        for simple geometric surfaces
        """
        if self.surface_type == 'sphere':
            return self._sphere_geodesic(point1, point2)
        elif self.surface_type == 'cylinder':
            return self._cylinder_geodesic(point1, point2)
        elif self.surface_type == 'torus':
            return self._torus_geodesic_approx(point1, point2)
        else:
            raise ValueError(f"Analytical solution not available for {self.surface_type}")
    
    def _sphere_geodesic(self, p1, p2):
        """Analytical geodesic distance on sphere surface"""
        # Normalize points to unit sphere
        n1 = p1 / np.linalg.norm(p1)
        n2 = p2 / np.linalg.norm(p2)
        
        # Calculate angular distance
        dot_product = np.clip(np.dot(n1, n2), -1, 1)
        angular_distance = np.arccos(dot_product)
        
        # Arc length = radius × angle
        return self.radius * angular_distance
    
    def _cylinder_geodesic(self, p1, p2):
        """Analytical geodesic distance on cylinder surface"""
        # Convert to cylindrical coordinates
        theta1 = np.arctan2(p1[1], p1[0])
        theta2 = np.arctan2(p2[1], p2[0])
        z1, z2 = p1[2], p2[2]
        
        # Handle angle wrapping
        dtheta = theta2 - theta1
        if dtheta > np.pi:
            dtheta -= 2 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2 * np.pi
        
        # Geodesic combines arc length and height difference
        arc_length = self.radius * abs(dtheta)
        height_diff = abs(z2 - z1)
        
        return np.sqrt(arc_length**2 + height_diff**2)
    
    def _torus_geodesic_approx(self, p1, p2):
        """Approximate geodesic distance on torus (simplified)"""
        # This is a simplified approximation
        # Exact solution requires elliptic integrals
        euclidean = self.euclidean_distance(p1, p2)
        
        # Rough approximation factor based on torus geometry
        curvature_factor = 1 + (self.minor_radius / self.major_radius) * 0.5
        return euclidean * curvature_factor
    
    def dijkstra_geodesic_distance(self, point1, point2, k_neighbors=8):
        """
        Calculate geodesic distance using Dijkstra's algorithm on mesh
        """
        if self.vertices is None:
            raise ValueError("No mesh available. Create a surface first.")
        
        # Find closest vertices to input points
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.vertices)
        _, idx1 = nbrs.kneighbors([point1])
        _, idx2 = nbrs.kneighbors([point2])
        start_idx, end_idx = idx1[0][0], idx2[0][0]
        
        # Build adjacency graph from mesh
        adj_matrix = self._build_mesh_adjacency_matrix(k_neighbors)
        
        # Run Dijkstra's algorithm
        distances = dijkstra(adj_matrix, indices=start_idx, return_predecessors=False)
        
        return distances[end_idx]
    
    def _build_mesh_adjacency_matrix(self, k_neighbors=8):
        """Build weighted adjacency matrix from mesh vertices"""
        n_vertices = len(self.vertices)
        
        # Use k-nearest neighbors to build local connections
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(self.vertices)
        distances, indices = nbrs.kneighbors(self.vertices)
        
        # Create sparse adjacency matrix
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(n_vertices):
            for j in range(1, k_neighbors):  # Skip self (j=0)
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                
                row_indices.extend([i, neighbor_idx])
                col_indices.extend([neighbor_idx, i])
                data.extend([distance, distance])
        
        adj_matrix = csr_matrix((data, (row_indices, col_indices)), 
                               shape=(n_vertices, n_vertices))
        return adj_matrix
    
    def fast_marching_geodesic(self, point1, point2, grid_resolution=100):
        """
        Approximate geodesic distance using Fast Marching Method
        This is a simplified implementation for demonstration
        """
        if self.vertices is None:
            raise ValueError("No mesh available. Create a surface first.")
        
        # Find closest vertices
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.vertices)
        _, idx1 = nbrs.kneighbors([point1])
        _, idx2 = nbrs.kneighbors([point2])
        start_idx, end_idx = idx1[0][0], idx2[0][0]
        
        # Simplified fast marching on mesh
        # In practice, this would solve the Eikonal equation numerically
        n_vertices = len(self.vertices)
        distances = np.full(n_vertices, np.inf)
        distances[start_idx] = 0
        
        # Priority queue for fast marching
        heap = [(0, start_idx)]
        visited = set()
        
        # Build neighbor connectivity
        adj_matrix = self._build_mesh_adjacency_matrix()
        
        while heap:
            current_dist, current_vertex = heapq.heappop(heap)
            
            if current_vertex in visited:
                continue
            if current_vertex == end_idx:
                break
                
            visited.add(current_vertex)
            
            # Update neighbors
            neighbors = adj_matrix[current_vertex].nonzero()[1]
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge_weight = adj_matrix[current_vertex, neighbor]
                    new_dist = current_dist + edge_weight
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
        
        return distances[end_idx]
    
    def compare_methods(self, point1, point2):
        """
        Compare all available geodesic distance calculation methods
        """
        results = {}
        
        # Euclidean distance (baseline)
        results['euclidean'] = self.euclidean_distance(point1, point2)
        
        # Analytical method (if available)
        try:
            results['analytical'] = self.analytical_geodesic_distance(point1, point2)
        except ValueError:
            results['analytical'] = None
        
        # Dijkstra method
        try:
            results['dijkstra'] = self.dijkstra_geodesic_distance(point1, point2)
        except:
            results['dijkstra'] = None
        
        # Fast Marching method
        try:
            results['fast_marching'] = self.fast_marching_geodesic(point1, point2)
        except:
            results['fast_marching'] = None
        
        return results
    
    def visualize_surface_and_points(self, point1, point2, title="3D Surface with Points"):
        """Visualize the 3D surface with the two points"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        if self.vertices is not None:
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 
                      alpha=0.6, s=1, c='lightblue')
        
        # Plot points
        ax.scatter(*point1, color='red', s=100, label='Point 1')
        ax.scatter(*point2, color='blue', s=100, label='Point 2')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()


def demonstrate_geodesic_calculations():
    """
    Comprehensive demonstration of geodesic distance calculations
    with stage-by-stage evaluation
    """
    print("=" * 80)
    print("3D SURFACE GEODESIC DISTANCE CALCULATOR - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Test cases for different surfaces
    test_cases = [
        {
            'surface': 'sphere',
            'params': {'radius': 2.0, 'resolution': 30},
            'points': [
                np.array([2.0, 0.0, 0.0]),      # Point on equator
                np.array([0.0, 2.0, 0.0])       # 90 degrees away
            ],
            'expected_geodesic': np.pi  # π radians = 90 degrees on unit sphere
        },
        {
            'surface': 'cylinder',
            'params': {'radius': 1.0, 'height': 4.0, 'resolution': 30},
            'points': [
                np.array([1.0, 0.0, -1.0]),     # Bottom of cylinder
                np.array([0.0, 1.0, 1.0])       # Top, 90 degrees around
            ],
            'expected_geodesic': None
        },
        {
            'surface': 'torus',
            'params': {'major_radius': 2.0, 'minor_radius': 0.5, 'resolution': 25},
            'points': [
                np.array([2.5, 0.0, 0.0]),      # Outer edge
                np.array([-2.5, 0.0, 0.0])      # Opposite side
            ],
            'expected_geodesic': None
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {case['surface'].upper()} SURFACE")
        print(f"{'='*60}")
        
        # Stage 1: Surface Generation
        print("\nSTAGE 1: Generating 3D Surface")
        print("-" * 40)
        
        calc = GeodesicDistanceCalculator()
        
        if case['surface'] == 'sphere':
            calc.create_sphere(**case['params'])
            print(f"✓ Sphere created: radius={case['params']['radius']}, "
                  f"vertices={len(calc.vertices)}")
        elif case['surface'] == 'cylinder':
            calc.create_cylinder(**case['params'])
            print(f"✓ Cylinder created: radius={case['params']['radius']}, "
                  f"height={case['params']['height']}, vertices={len(calc.vertices)}")
        elif case['surface'] == 'torus':
            calc.create_torus(**case['params'])
            print(f"✓ Torus created: major_radius={case['params']['major_radius']}, "
                  f"minor_radius={case['params']['minor_radius']}, vertices={len(calc.vertices)}")
        
        # Stage 2: Point Selection and Validation
        print("\nSTAGE 2: Point Selection and Surface Projection")
        print("-" * 50)
        
        point1, point2 = case['points']
        print(f"Point 1: ({point1[0]:.3f}, {point1[1]:.3f}, {point1[2]:.3f})")
        print(f"Point 2: ({point2[0]:.3f}, {point2[1]:.3f}, {point2[2]:.3f})")
        
        # Verify points are on surface (approximately)
        if case['surface'] == 'sphere':
            dist1_to_center = np.linalg.norm(point1)
            dist2_to_center = np.linalg.norm(point2)
            print(f"✓ Point 1 distance from center: {dist1_to_center:.3f} "
                  f"(expected: {calc.radius:.3f})")
            print(f"✓ Point 2 distance from center: {dist2_to_center:.3f} "
                  f"(expected: {calc.radius:.3f})")
        
        # Stage 3: Distance Calculations
        print("\nSTAGE 3: Computing Distances Using Multiple Methods")
        print("-" * 55)
        
        results = calc.compare_methods(point1, point2)
        
        print(f"Euclidean Distance:    {results['euclidean']:.6f} units")
        
        if results['analytical'] is not None:
            ratio_analytical = results['analytical'] / results['euclidean']
            print(f"Analytical Geodesic:   {results['analytical']:.6f} units "
                  f"(ratio: {ratio_analytical:.3f})")
            
            if case['expected_geodesic'] is not None:
                error = abs(results['analytical'] - case['expected_geodesic'])
                print(f"  → Expected:          {case['expected_geodesic']:.6f} units "
                      f"(error: {error:.6f})")
        
        if results['dijkstra'] is not None:
            ratio_dijkstra = results['dijkstra'] / results['euclidean']
            print(f"Dijkstra Geodesic:     {results['dijkstra']:.6f} units "
                  f"(ratio: {ratio_dijkstra:.3f})")
        
        if results['fast_marching'] is not None:
            ratio_fm = results['fast_marching'] / results['euclidean']
            print(f"Fast Marching:         {results['fast_marching']:.6f} units "
                  f"(ratio: {ratio_fm:.3f})")
        
        # Stage 4: Method Comparison and Analysis
        print("\nSTAGE 4: Method Comparison and Analysis")
        print("-" * 45)
        
        methods = ['analytical', 'dijkstra', 'fast_marching']
        valid_methods = {k: v for k, v in results.items() 
                        if k in methods and v is not None}
        
        if len(valid_methods) > 1:
            method_names = list(valid_methods.keys())
            method_values = list(valid_methods.values())
            
            # Find method closest to analytical (if available)
            if 'analytical' in valid_methods:
                reference = valid_methods['analytical']
                print(f"Using analytical solution as reference: {reference:.6f}")
                
                for method, value in valid_methods.items():
                    if method != 'analytical':
                        error = abs(value - reference)
                        error_pct = (error / reference) * 100
                        print(f"  → {method} error: {error:.6f} ({error_pct:.2f}%)")
            else:
                # Compare all methods to each other
                print("Comparing numerical methods:")
                for i, (m1, v1) in enumerate(valid_methods.items()):
                    for j, (m2, v2) in enumerate(valid_methods.items()):
                        if i < j:
                            diff = abs(v1 - v2)
                            avg = (v1 + v2) / 2
                            diff_pct = (diff / avg) * 100
                            print(f"  → {m1} vs {m2}: {diff:.6f} difference "
                                  f"({diff_pct:.2f}%)")
        
        # Visualization (optional - uncomment to show plots)
        # calc.visualize_surface_and_points(point1, point2, 
        #                                   f"{case['surface'].title()} Surface")
    
    # Additional demonstration: Effect of mesh resolution
    print(f"\n{'='*80}")
    print("MESH RESOLUTION ANALYSIS")
    print(f"{'='*80}")
    
    resolutions = [10, 20, 30, 50]
    sphere_point1 = np.array([1.0, 0.0, 0.0])
    sphere_point2 = np.array([0.0, 1.0, 0.0])
    analytical_expected = np.pi / 2  # 90 degrees on unit sphere
    
    print(f"\nAnalyzing effect of mesh resolution on geodesic calculation:")
    print(f"Surface: Unit sphere, Points separated by 90 degrees")
    print(f"Expected analytical geodesic distance: {analytical_expected:.6f}")
    print("-" * 60)
    
    for res in resolutions:
        calc = GeodesicDistanceCalculator()
        calc.create_sphere(radius=1.0, resolution=res)
        
        dijkstra_dist = calc.dijkstra_geodesic_distance(sphere_point1, sphere_point2)
        error = abs(dijkstra_dist - analytical_expected)
        error_pct = (error / analytical_expected) * 100
        
        print(f"Resolution {res:2d}: {dijkstra_dist:.6f} units "
              f"(error: {error:.6f}, {error_pct:.2f}%)")


def performance_benchmark():
    """Benchmark different methods for performance comparison"""
    print(f"\n{'='*80}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*80}")
    
    import time
    
    # Create test surface
    calc = GeodesicDistanceCalculator()
    calc.create_sphere(radius=1.0, resolution=50)
    
    point1 = np.array([1.0, 0.0, 0.0])
    point2 = np.array([0.0, 1.0, 0.0])
    
    methods = {
        'Analytical': lambda: calc.analytical_geodesic_distance(point1, point2),
        'Dijkstra': lambda: calc.dijkstra_geodesic_distance(point1, point2),
        'Fast Marching': lambda: calc.fast_marching_geodesic(point1, point2)
    }
    
    n_iterations = 10
    print(f"Running {n_iterations} iterations for each method...")
    print("-" * 50)
    
    for method_name, method_func in methods.items():
        times = []
        
        for _ in range(n_iterations):
            start_time = time.time()
            try:
                result = method_func()
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"{method_name}: Error - {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{method_name:15s}: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")


if __name__ == "__main__":
    # Run comprehensive demonstration
    demonstrate_geodesic_calculations()
    
    # Run performance benchmark
    performance_benchmark()
    
    print(f"\n{'='*80}")
    print("SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*80}")
