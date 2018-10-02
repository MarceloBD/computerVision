from projecao import Projecao


if __name__ == '__main__':

	Tc = [-1000 ,1000 ,5000]
	roty = 160
	f = 16
	sx, sy = [0.01,0.01]
	ox = 320
	oy = 240 

	Pw = [[1000,1000,1500,1500], 
		  [1000,1500,1500,1000], 
		  [ 500,500, 500, 500]]


	proj = Projecao()
	h = proj.homogenea(0, roty, 0, Tc[0], Tc[1], Tc[2])
	Pc = proj.mundo_para_camera(Pw,h)
	pj = proj.proj_perspectiva_mm(Pc, f)

	pjp = proj.proj_perspectiva_pixel(pj, sx, sy, ox, oy)

	proj.printMatrix(pjp)