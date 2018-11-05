from projecao import Projecao
import numpy as np

if __name__ == '__main__':

	proj = Projecao()

	### camera 1 direita
	pontos = proj.pontos_calibracao(10,10,1)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0 , 0, 0, 5, 5 ,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	rr = np.array(H)[:,0:3]
	tr = np.array(H)[:,3]

	Pc = proj.mundo_para_camera(pontos,H)

	pj = proj.proj_perspectiva_mm(Pc, f)
	
	print(Pc[:,10:14])
	print(pj[:,10:14])
	pontos_proj_r,zr = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj_r)

	proj.calcular_matriz_essencial(H, dx, dy, dz)
	### camera 2 esquera
	
	pontos = proj.pontos_calibracao(10,10,1)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0 , 0, 0, 10, 10 ,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	rl = np.array(H)[:,0:3]
	tl = np.array(H)[:,3]

	Pc = proj.mundo_para_camera(pontos,H)

	
	pj = proj.proj_perspectiva_mm(Pc, f)
	
	print('pc', Pc[:,10:13], 'pj', pj[:,10:13])


	pontos_proj_l, zl = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj_l)

	####
	e =proj.calcular_matriz_essencial(H, dx, dy, dz)
	print('e', e)

	fund = proj.calcular_matriz_fundamental(pontos_proj_l[:,0:9], pontos_proj_r[:,0:9], e)
	print('f', fund)
	fund_est = proj.estimar_matriz_fundamental(pontos_proj_l[:,0:9], pontos_proj_r[:,0:9], e)
	print('f_est', fund_est)
	fund = fund_est

	print('epipolo_esquerda', proj.epipolo_esquerda(fund))
	print('epipolo_direita', proj.epipolo_direita(fund))


	matrix = pontos_proj_l
	for i in np.arange(0,300,10):
		[xl, yl] = proj.linha_epipolar_esquerda(fund, pontos_proj_l[:,[i,i+1]])
		col = np.vstack((xl, yl))
		matrix = np.hstack((matrix, col))

	proj.print_in_screen(matrix)



	matrix = pontos_proj_r
	for i in np.arange(0,300,10):
		[xr, yr] = proj.linha_epipolar_direita(fund, pontos_proj_r[:,[i,i+1]])
		col = np.vstack((xr, yr))
		matrix = np.hstack((matrix, col))

	proj.print_in_screen(matrix)
	


	r = proj.achar_r_triangularizacao(rl, rr)
	t = proj.achar_t_triangularizacao(r, tl, tr)

	print('r',r,'t',t)

	a,b,c = proj.achar_abc_triangularizacao(pontos_proj_l[:,10], pontos_proj_r[:,10],r, t)

	w = proj.achar_w_triangularizacao(pontos_proj_l[:,10], pontos_proj_r[:,10], r)

	pontos_proj_l = np.append(pontos_proj_l,  [100*16*np.ones(len(pontos_proj_r[0]))], axis=0)

	print('wl',a*pontos_proj_l[:,12]+c*w)
	pontos_proj_r = np.append(pontos_proj_r, [100*16*np.ones(len(pontos_proj_r[0]))], axis=0)
	print('wr',t+b*np.matmul(np.matrix.transpose(r),pontos_proj_r[:,12]))