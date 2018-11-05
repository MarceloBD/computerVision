from copy import copy, deepcopy
import math 
import numpy as np
import cv2 as cv2


class Projecao:

	##########################################################
	def proj_perspectiva_mm(self, P, f):
	# @P matriz 3xN em [mm]
	# @f distancia focal da camera [mm]
	# retorna projecao do ponto P [mm] 
		p = copy(P)
		for i in range(len(P[0])):
			p[0][i] = f*P[0][i]/float(P[2][i])
			p[1][i] = f*P[1][i]/float(P[2][i])
			p[2][i] = P[2][i]/float(f)
		return p

	def proj_perspectiva_pixel(self, p, sx, sy, ox, oy):
	# @p coordenadas a projecao do pontos [mm]
	# @sx dimensao horizontal do pixel [mm/pixel]
	# @sy dimensao vertical do pixel [mm/pixel]
	# @ox coordenada de projecao do eixo optico [pixel]
	# retorna coordenada do pixel [pixel]
		z = pixel = np.zeros(shape=(1, len(p[0]))) 
		pixel = np.zeros(shape=(2, len(p[0])))
		for i in range(len(p[0])):
			pixel[0][i] = -int(p[0][i]/sx + ox) 
			pixel[1][i] = -int(p[1][i]/sy + oy)
			z[0][i] = p[2][i]
		return pixel, z

	def printMatrix(self, matrix):
		fmt ='{:>4}'
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				value = '{:5.1f}'.format(matrix[i][j])
				print (fmt.format(value), end=" ")
			print("")			

	###########################################################


	def mundo_para_camera(self,Pw,H):
	# Pw matriz 3xN [mm] coordenada global
	# H matriz de transformacao homogenea [mm]
	# retorna ponto em coordenada da camera 3xN[mm]
		
		P = copy(Pw)
		Pc = copy(Pw)
		P = np.vstack((Pw,[1 for _ in range(len(Pw[0]))]))
		for n in range(len(P[0])):
			Pc[0][n], Pc[1][n], Pc[2][n] = [0,0,0]
			for i in range(4):
				Pc[0][n] += H[0][i]*P[i][n]
				Pc[1][n] += H[1][i]*P[i][n]
				Pc[2][n] += H[2][i]*P[i][n]
		return Pc

	def homogenea(self,rotx, roty, rotz, dx, dy, dz):
	# rotx, roty, rotz rotacao em graus 
	# dx, dy, dz deslocamento em [mm]
	# retorna matriz de transformacao homogenea (3xN)
		pi = np.pi
		rotxR = rotx*pi/180.0  
		rotyR = roty*pi/180.0 
		rotzR = rotz*pi/180.0  
		cz = np.cos(rotzR) 
		sz = np.sin(rotzR)
		cy = np.cos(rotyR) 
		sy = np.sin(rotyR)
		cx = np.cos(rotxR) 
		sx = np.sin(rotxR)

		H = [[cz*cy, -1*sz*cx+cz*sy*sx, sz*sx+cz*sy*cx,  -1*dx],
			 [sz*cy, cz*cx+sz*sy*sx,  -cz*sx+sz*sy*cx, -1*dy],
			 [-1*sy,   cy*sx,           cx*cy,           -1*dz]]
		return H

	##################################################################

	def pontos_plano(self,n,m,d):
	# n linha e impar 
	# m coluna e impar
	# d distancia entre linhas e colunas [mm]
		length = n*m
		Pp =  [[1 for _ in range(length)],
			   [1 for _ in range(length)],
			   [1 for _ in range(length)]]
		first_col = -((m+1)/2 - 1)*d
		col_val = first_col
		first_lin = -((n+1)/2 - 1)*d
		lin_val = first_lin
		for col in range(m):
			for lin in range(n):
				Pp[0][col+lin*m] = col_val
			col_val += d
		for point in range(length):
			lin_val = first_lin + (int(point/m))*d 
			Pp[1][point] = lin_val
			Pp[2][point] = 0
		return Pp


	def print_in_screen(self, matrix):
		img = np.zeros((600, 600, 3), np.uint8)
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				matrix[i][j] = abs(matrix[i][j])
		for i in range(len(matrix[0])):
			img = cv2.circle(img, tuple(map(int, matrix[:, i])), 3, (255, 255, 255))
		cv2.imwrite('rectangle.jpg', img)
		cv2.imshow('img', img)
		k = cv2.waitKey(0) 

	######################################################################

	def fazer_pontos_positivos(self, vector, n, m,d):
		dx, dy, dz = [-((m+1)/2 - 1)*d, -((n+1)/2 - 1)*d, 0]
		rotx, roty, rotz = [0, 0, 0]
		h = self.homogenea(rotx, roty, rotz, dx, dy, dz)
		return self.mundo_para_camera(vector, h)

	def pontos_calibracao(self, n, m, d):
		p = self.pontos_plano(n, m, d)
		#self.printMatrix(p)
		print()
		dx, dy, dz = [0, 0, 0]
		rotx, roty, rotz = [0, 0, 0]
		px = self.fazer_pontos_positivos(p, n, m,d)
		p_calib = px 

		rotx, roty, rotz = [0, -90, 0]
		h = self.homogenea(rotx, roty, rotz, dx, dy, dz)
		py = self.mundo_para_camera(deepcopy(px), h)
		p_calib = np.hstack((p_calib, py))		
 
		rotx, roty, rotz = [90, 0, 0]
		h = self.homogenea(rotx, roty, rotz, dx, dy, dz)
		pz = self.mundo_para_camera(deepcopy(px), h)
		p_calib = np.hstack((p_calib, pz))

		return p_calib

	###################################################################################################################


	def resolver_Amb(self, A):
		A_transposta = np.matrix.transpose(A)
		u, s, vt =  np.linalg.svd(np.matmul(A_transposta,A)) 
		v = np.matrix.transpose(vt)
		m = v[:,len(v[0])-1]
		return m

	def construir_A(self, Pcal, Ical):
		A =  np.empty((0,12), float)
		for i in range(len(Pcal[0])):
			x = Ical[0, i]
			y = Ical[1, i]
			Xw = Pcal[0, i] 
			Yw = Pcal[1, i] 
			Zw = Pcal[2, i]
			A_line1 = np.array([[Xw, Yw, Zw, 1, 
								0, 0, 0, 0,
								-x*Xw, -x*Yw, -x*Zw, -x]])
			A_line2 = np.array([[0, 0, 0, 0,
								Xw, Yw, Zw, 1, 
								-y*Xw, -y*Yw, -y*Zw, -y]])
			A = np.append(A, A_line1, axis=0)
			A = np.append(A, A_line2, axis=0)
		return A 

	def calibracao(self, Pcal, Ical):
		A = self.construir_A(Pcal, Ical)
		
		m = self.resolver_Amb(A)

		gamma_abs = np.sqrt(m[8]**2+ m[9]**2+ m[10]**2)

		for i in range(12):
			m[i] = m[i]/gamma_abs

		[q1, q2, q3, q4] = [ [m[0], m[1], m[2]],
						   [m[4], m[5], m[6]],
						   [m[8], m[9], m[10]],
						   [m[3], m[7], m[11]]]
		ox = np.array(q1).dot(q3)
		oy = np.array(q2).dot(q3)
		fx = np.sqrt(np.array(q1).dot(q1)-ox**2)
		fy = np.sqrt(np.array(q2).dot(q2)-oy**2)
		print('ox oy fx fy',ox, oy, fx, fy)


		r_matrix = [[(m[0]-ox*m[8])/(-fx), (m[1]-ox*m[9])/(-fx), (m[2]-ox*m[10])/(-fx)],
					[(m[4]-oy*m[8])/(-fy), (m[5]-oy*m[9])/(-fy), (m[6]-oy*m[10])/(-fy)],
					[m[8], m[9], m[10]]]


		print('r_matrix', r_matrix)
		T = [ (m[3]-ox*m[11])/(-fx), (m[7]-oy*m[11])/(-fy), m[11]]

		print(T)
		return [ox, oy, fx, fy, r_matrix, T]

	##############################################################################################

	def calcular_matriz_essencial(self, h, tx, ty, tz):
		r = np.array(h)[0:3,0:3]
		s = [[0, -tz, -ty],
			[tz, 0, -tx],
			[-ty, tx, 0]]
		return np.matmul(r,s)

	def estimar_matriz_fundamental(self, pl, pr,e ):
		A =  np.empty((0,9), float)
		for p in range(len(pl[0])):
			A_line = np.array([[pl[0][p]*pr[0][p], pl[0][p]*pr[1][p], pl[0][p], pl[1][p]*pr[0][p], pl[1][p]*pr[1][p], pl[1][p], pr[0][p], pr[1][p], 1]])
			A = np.append(A, A_line, axis=0)
		f = self.resolver_Amb2(A)
		f = f.reshape(3,3)
#		m = [[16, 0, 0],[0,16,0],[0,0,1]]
#		return np.matmul(np.linalg.inv(m), np.matmul(e,np.linalg.inv(m)))
		return f
	
	def calcular_matriz_fundamental(self, pl, pr,e ):
		m = [[16/0.01, 0, 0],[0,16/0.01,0],[0,0,1]]
		return np.matmul(np.linalg.inv(m), np.matmul(e,np.linalg.inv(m)))
		return f

	def resolver_Amb2(self, A):
		u, s, vt =  np.linalg.svd(A) 
		v = np.matrix.transpose(vt)
		m = v[:,len(v[0])-1]
		return m	

	def epipolo_esquerda(self, f):
		[d, v] = np.linalg.eig(np.matmul(f, np.matrix.transpose(f))) 
		e = v[:,0]/v[2,0]
		#e = self.resolver_Amb2(f)	
		return e

	def epipolo_direita(self, f):
		[d, v] = np.linalg.eig(np.matmul(np.matrix.transpose(f), f)) 
		e = v[:,0]/v[2,0]
		#A_transposta = np.matrix.transpose(f)
		#u, s, vt =  np.linalg.svd(f) 
		#e = u[:,len(u[0])-1]
		return e	

	def linha_epipolar_esquerda(self, f, pl):
		l = np.empty((0,3), float)
		pl = np.vstack([pl, np.ones(len(pl[0]))])
		for index in range(len(pl[0])):
			l = np.append(l, np.matmul(f, pl[:,index]))
			l = l/l[2]
		x = np.arange(600) 
		y = -l[0]/l[1]*x- l[2]/l[1]
		return x, y

	def linha_epipolar_direita(self, f, pr):
		l = np.empty((0,3), float)
		pr = np.vstack([pr, np.ones(len(pr[0]))])
		for index in range(len(pr[0])):
			l = np.append(l, np.matmul(pr[:,index], f))
			l = l/l[2]
		x = np.arange(600) 
		y = -l[0]/l[1]*x - l[2]/l[1]
		return x, y


	def achar_o_camera(self, h):
		o = self.mundo_para_camera([[0],[0],[0]],h)
		return o

	def achar_w_triangularizacao(self, pl, pr, r):
		pl = np.append(pl, -6.25)
		pr = np.append(pr, -6.25)
		return np.cross(pl, np.matmul(np.matrix.transpose(r), pr))

	def achar_r_triangularizacao(self, rl, rr):
		return np.matmul(rr, np.matrix.transpose(rl))

	def achar_t_triangularizacao(self, r, tl, tr):
		return tl - np.matmul(np.matrix.transpose(r), tr)

	def achar_abc_triangularizacao(self, pl, pr, r, t):
		pl = np.append(pl, -6.25)
		pr = np.append(pr, -6.25)
		rt = np.matrix.transpose(r)
		A = np.empty((0,3), float)
		for i in range(3):
			line = [[pl[i], -np.matmul(rt,pr)[i], np.cross(pl, np.matmul(rt, pr))[i]]]
			A = np.append(A, line, axis=0)
		abc = np.linalg.solve(A,t)
		print(abc)
		return abc 