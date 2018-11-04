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
	pontos_proj_r = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
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
	pontos_proj_l = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj_l)

	####
	proj.calcular_matriz_essencial(H, dx, dy, dz)

	fund = proj.calcular_matriz_fundamental(pontos_proj_l[:,0:9], pontos_proj_r[:,0:9])

	print('epipolo_esquerda', proj.epipolo_esquerda(fund))
	print('epipolo_direita', proj.epipolo_direita(fund))

	proj.linha_epipolar_esquerda(fund, pontos_proj_l[:,[0,99,299]])

	r = proj.achar_r_triangularizacao(rl, rr)
	t = proj.achar_t_triangularizacao(r, tl, tr)

	print('r',r,'t',t)

	a,b,c = proj.achar_abc_triangularizacao(pontos_proj_l[:,10], pontos_proj_r[:,10],r, t)

	print('wl',a*pontos_proj_l[:,10:14])
	pontos_proj_r = np.append(pontos_proj_r, [np.ones(len(pontos_proj_r[0]))], axis=0)
	print('wr',t+b*np.matmul(np.matrix.transpose(r),pontos_proj_r[:,12]))