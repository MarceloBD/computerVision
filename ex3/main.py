from projecao import Projecao


if __name__ == '__main__':

	proj = Projecao()

	pontos = proj.pontos_calibracao(4,4,4)

	proj.printMatrix(pontos)

	[f, sx, sy, ox, oy] = [16, 0.01, 0.01, 320, 240]
	rotx, roty, rotz, dx, dy, dz = [0,0,0, 100, 100,100]
	
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(pontos,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pontos_proj = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.print_in_screen(pontos_proj)

	proj.calibracao(pontos, pontos_proj)