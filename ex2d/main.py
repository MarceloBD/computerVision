from projecao import Projecao


if __name__ == '__main__':
	

	f = 16
	sx, sy = [0.01,0.01]
	ox, oy = [320,240]
	n, m, d = [3,3,10]

	proj = Projecao()
	Pw = proj.pontos_plano(n,m,d)
	proj.printMatrix(Pw)
	#1) 
	print("1)")
	rotx, roty, rotz = [180, 0, 0]
	dx, dy, dz = [0,0,500]

	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(Pw,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.printMatrix(pjp)

	print("2)")
	rotx, roty, rotz = [180, 0, 0]
	dx, dy, dz = [0,0,1000]
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(Pw,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.printMatrix(pjp)

	print("3)")
	rotx, roty, rotz = [180+45, 0, 0]
	dx, dy, dz = [0,0,1000]
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(Pw,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.printMatrix(pjp)

	print("4)")
	rotx, roty, rotz = [180+45, 0, 0]
	dx, dy, dz = [0,0,1500]
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(Pw,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.printMatrix(pjp)

	print("5)")
	rotx, roty, rotz = [180+45, 0, 0]
	dx, dy, dz = [0,0,2000]
	H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
	Pc = proj.mundo_para_camera(Pw,H)
	pj = proj.proj_perspectiva_mm(Pc, f)
	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)
	proj.printMatrix(pjp)
