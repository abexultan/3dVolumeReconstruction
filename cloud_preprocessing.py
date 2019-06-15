from open3d import np, PointCloud, Vector3dVector, read_point_cloud, voxel_down_sample
import glob, os


def frompcd_to_xyz(path_to_ply):
    pcd = read_point_cloud(path_to_ply)
    xyz_load = np.asarray(pcd.points)*25 #upscale the sparsity of pcd
    minx = min(xyz_load[:,0])
    miny = min(xyz_load[:,1])
    minz = min(xyz_load[:,2])
    xyz_load_n = np.zeros(xyz_load.shape, dtype=int)
    xyz_load_n[:,0] = xyz_load[:,0] - minx 
    xyz_load_n[:,1] = xyz_load[:,1] - miny
    xyz_load_n[:,2] = xyz_load[:,2] - minz
    xyz_64 = xyz_load_n.round() 
    return xyz_64

if __name__ == '__main__':
	names = ["akmaral", "albina", "alen", "alfarabi", "askhat", "beka", "bex", "denis", "lashin", "lera", "mika", "nazerke", "shamil", "zhanat", "zhanel"]
	for n in names:
	    for l in range(1,4):
	        for k in range(1,11):
	            path = "/media/bexultan/DATA_STORAGE/clouds/"+n+"_day"+str(l)+"_clouds/"+n+"_"+str(k)+".bag_clouds/"
	            f1 = []
	            os.chdir(path)
	            for file in glob.glob("cloud1_*"):
	                f1.append(file[7:-5])
	            f1 = list(map(int, f1))
	            f1.sort()
	            f2 = []
	            for file in glob.glob("cloud2_*"):
	                f2.append(file[7:-5])
	            f2 = list(map(int, f2))
	            f2.sort()
	            f3 = []
	            for file in glob.glob("cloud3_*"):
	                f3.append(file[7:-5])
	            f3 = list(map(int, f3))
	            f3.sort()
	            f4 = []
	            for file in glob.glob("cloud4_*"):
	                f4.append(file[7:-5])
	            f4 = list(map(int, f4))
	            f4.sort()
	            f_result = list(set(f1) & set(f2) & set(f3) & set(f4))
	            f_result.sort()
	            for i in range(1,5):
	                for j in range(300):
	                    if(j in f_result):
	                        temp_x = frompcd_to_xyz(path+"cloud"+str(i)+"_"+str(j)+"0.ply")
	                        temp_y = frompcd_to_xyz(path+"cloud"+"_"+"merged"+"_"+str(j)+"0.ply")
	                        max_x = format(np.max(temp_y[:,0]), '03d')
	                        max_y = format(np.max(temp_y[:,1]), '03d')
	                        max_z = format(np.max(temp_y[:,2]), '03d')
	                        np.savetxt(path+n+"_scenario"+str(k)+"_clouds64_25d/voxel_grids_64/"+"cloud"+str(i)+"_"+str(j)+'_'+max_x+"_"+max_y+"_"+max_z+".txt",temp_x, fmt='%i')
	                        np.savetxt(path+n+"_scenario"+str(k)+"_clouds64_3d/voxel_grids_64/"+"cloud"+str(i)+"_"+str(j)+'_'+max_x+'_'+max_y+'_'+max_z+".txt",temp_y, fmt='%i')
	            print("Saved "+n+" day"+str(l)+" scenario "+str(k))
	        print("Finished " +n+ " day "+str(l))