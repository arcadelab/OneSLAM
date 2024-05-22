import numpy as np
import g2o


"""
BA optimizer adapted from g2opy GitHub page.
INFO: Could not find nice documentation for everything you can do in g2opy.
      Best way I found was to use built-in python functions to look up available functions.
"""
class BundleAdjuster(g2o.SparseOptimizer):
    def __init__(self, use_sparse_solver=False):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        if use_sparse_solver:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, intrinsics, fixed=False):
        fx, fy, cx, cy = intrinsics

        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(fx, fy, cx, cy, 0)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
    
    def add_pose_edge(self, pose_indicies, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(pose_indicies):
            if isinstance(v, int):
                v = self.vertex(v*2)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        
        super().add_edge(edge)
    
    

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

        return edge

    def update_edge(self, edge, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI
        
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)


    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()


    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()
    
    def fix_pose(self, pose_id, fixed=True):
        self.vertex(pose_id * 2).set_fixed(fixed)
    
    def fix_point(self, point_id, fixed=True):
        self.vertex(point_id * 2 + 1).set_fixed(fixed)