from Datasets.transformation import kitti2tartan, pos_quats2SE_matrices, SE2pos_quat
from evaluator.evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, transform_trajs, quats2SEs
from evaluator.trajectory_transform import timestamp_associate
from Datasets.utils import plot_traj
from evaluator.evaluate_ate_scale import align
from evaluator.tartanair_evaluator import TartanAirEvaluator
import numpy as np

"""---THIS IS FOR KITTI ONLY---"""
"""
path = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/KITTI/10.txt"
traj = np.loadtxt(path, dtype=float) #this loads the text that's within the file 
x = kitti2tartan(traj)
np.savetxt("/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/kitti2tartan/10.txt", x)

evaluator = TartanAirEvaluator() #this is the evaluator that we will use 

#Put GROUND TRUTH DATA IN HERE
gt_traj = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/kitti2tartan/10.txt" 

#FOR KITTI, PUT THE TEXT FILE YOU SAVED UP THERE
est_traj = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/KITTI/Trajectories/10.txt"


results = evaluator.evaluate_one_trajectory(gt_traj, est_traj,scale=True, kittitype=True)
print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))


plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/tartanvo-main/results/results_K10.png', title='ATE %.4f' %(results['ate_score']))
np.savetxt('/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/tartanvo-main/results/results_K10.txt',results['est_aligned'])
"""


"""THIS IS FOR EUROC ONLY"""
#"""
path = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/EUROC/V2_02.csv"

traj = np.loadtxt(path, delimiter=",",usecols=range(1, 8), skiprows=1)
#traj = traj[::10]
#traj = traj[:-1]

np.savetxt("/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/EUROC_SPLICED/V2_02.txt", traj)
#need to splice the data

evaluator = TartanAirEvaluator() #this is the evaluator that we will use 
'---PUT GROUND TRUTH HERE---'

gt_traj = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/data/EUROC_SPLICED/V2_02.txt" 
est_traj = "/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/EuRoC/Trajectories/V_202.txt"
gt_xyz = np.matrix(gt_traj[:,0:3].transpose())
est_xyz = np.matrix(est_traj[:, 0:3].transpose())

rot, trans, trans_error, s = align(gt_xyz, est_xyz, 1)
print('  ATE scale: {}'.format(s))
error = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
est_SEs = pos_quats2SE_matrices(est_traj)
T = np.eye(4) 
T[:3,:3] = rot
T[:3,3:] = trans 
T = np.linalg.inv(T)
est_traj_aligned = []
for se in est_SEs:
    se[:3,3] = se[:3,3] * s
    se_new = T.dot(se)
    se_new = SE2pos_quat(se_new)
    est_traj_aligned.append(se_new)

est_traj_aligned = np.array(est_traj_aligned)

results = evaluator.evaluate_one_trajectory(gt_traj, est_traj, scale=True, kittitype=False)
print("==> ATE: %.4f" %(results['ate_score']))
print("==> RPE: ")
print(results['rpe_score'])
#print("==> RPE: %.4f" %(results['rpe_score']))
plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/tartanvo-main/results/results_V202.png', title='ATE %.4f' %(results['ate_score']))
np.savetxt('/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/tartanvo-main/results/results_V202.txt',results['est_aligned'])
#np.savetxt('/Users/krisadhi/Library/CloudStorage/OneDrive-UniversityofMassachusetts/fall2023/635/tartanvo-main/results/resultsrpe_V202.txt',rpe_score)
#"""
