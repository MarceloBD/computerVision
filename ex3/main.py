from projecao import Projecao
import numpy as np

if __name__ == '__main__':

	proj = Projecao()

	pontos = proj.pontos_calibracao(10,10,1)

	#proj.printMatrix(pontos)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0 , 0, 0, 5, 5 ,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	print(pontos.shape)
	print(np.array(H).shape)
	Pc = proj.mundo_para_camera(pontos,H)

	pj = proj.proj_perspectiva_mm(Pc, f)
	pontos_proj = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj)

	[ox, oy, fx, fy, r_matrix, T] = proj.calibracao(pontos, pontos_proj)

	##############
	[f, sx, sy, ox, oy] = [fx*0.01, 0.01, 0.01, ox, oy]


	r_matrix = np.array(r_matrix)
	r_matrix = np.column_stack((r_matrix, [-T[0], -T[1], T[2]]))
	r_matrix[0,0] = r_matrix[0,0]*-1
	r_matrix[1,1] = r_matrix[1,1]*-1
	r_matrix[0,2] = r_matrix[0,2]*-1
	r_matrix[1,2] = r_matrix[1,2]*-1
	r_matrix[2,0] = r_matrix[2,0]*-1
	r_matrix[2,1] = r_matrix[2,1]*-1

	print('-')
	proj.printMatrix(r_matrix)
	print()
	proj.printMatrix(H)

	Pc = proj.mundo_para_camera(pontos,np.array(r_matrix))

	pj = proj.proj_perspectiva_mm(Pc, f)
	pontos_proj = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj)