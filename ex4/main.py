from projecao import Projecao
import numpy as np

if __name__ == '__main__':

	proj = Projecao()

	### camera 1 direita
	pontos = proj.pontos_calibracao(10,10,1)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0 , 0, 0, 5, 5 ,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)

	Pc = proj.mundo_para_camera(pontos,H)

	pj = proj.proj_perspectiva_mm(Pc, f)
	pontos_proj_r = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj_r)


	print(proj.calcular_matriz_essencial(H, dx, dy, dz))
	### camera 2 esquera
	
	pontos = proj.pontos_calibracao(10,10,1)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0 , 0, 0, 10, 10 ,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)

	Pc = proj.mundo_para_camera(pontos,H)

	pj = proj.proj_perspectiva_mm(Pc, f)
	pontos_proj_l = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj_l)

	####
	print(proj.calcular_matriz_essencial(H, dx, dy, dz))

	proj.calcular_matriz_fundamental(pontos_proj_l[:,0:9], pontos_proj_r[:,0:9])