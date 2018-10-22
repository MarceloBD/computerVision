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
			p[2][i] = f
		return p

	def proj_perspectiva_pixel(self, p, sx, sy, ox, oy):
	# @p coordenadas a projecao do pontos [mm]
	# @sx dimensao horizontal do pixel [mm/pixel]
	# @sy dimensao vertical do pixel [mm/pixel]
	# @ox coordenada de projecao do eixo optico [pixel]
	# retorna coordenada do pixel [pixel]
		pixel = np.zeros(shape=(2, len(p[0])))
		for i in range(len(p[0])):
			pixel[0][i] = -int(p[0][i]/sx + ox) 
			pixel[1][i] = -int(p[1][i]/sy + oy)
		return pixel

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
		self.printMatrix(p)
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


	#	hx = self.homogenea(rotx, roty, rotz, dx, dy, dz)
	#	px = self.mundo_para_camera(p, hx)

	###########################################################
	'''
	def solucao_linear(self, A, b):
		A_transposta = np.matrix.transpose(A)
		u, s, v =  np.linalg.svd(np.matmul(A_transposta,A)) 
	
		x = v[:,len(v[0])-1]
		#pseudo_inversa = np.matmul(np.linalg.inv(s), np.matrix.transpose(u))
		#pseudo_inversa = np.linalg.pinv(np.matmul(A_transposta, A))
		#x = np.matmul(pseudo_inversa, A_transposta).dot(b)
		return x

	def achar_lambda_alpha_v(self, Pcal, pontos_proj):
		A= np.empty((0,8), float)

		for i in range(len(Pcal[0])):
			x = pontos_proj[0, i]
			y = pontos_proj[1, i]
			Xw = Pcal[0, i] 
			Yw = Pcal[1, i] 
			Zw = Pcal[2, i] 
			A_line = np.array([[x*Xw, x*Yw, x*Zw, x, -y*Xw, -y*Yw, -y*Zw, -y]])
			A = np.append(A, A_line, axis=0) 
		v_solucao = self.solucao_linear(A, np.zeros(len(Pcal[0])))

		lambda_abs = np.sqrt(v_solucao[0]**2+v_solucao[1]**2+v_solucao[2]**2)
		alpha = np.sqrt(v_solucao[4]**2+v_solucao[5]**2+v_solucao[6]**2)/lambda_abs
		return [lambda_abs, alpha, v_solucao]
	
	def solucao_axb(self, A, b):
		A_transposta = np.matrix.transpose(A)
		pseudo_inversa = np.linalg.pinv(np.matmul(A_transposta, A))
		return (np.matmul(pseudo_inversa, A_transposta).dot(b))

	def achar_tz_fx(self, Pcal, pontos_proj, r_matrix):
		A =  np.empty((0,2), float)
		b = np.empty((0,1), float)
		Tx = 0 
		for i in range(len(Pcal)):
			x = pontos_proj[0, i]
			Xw = Pcal[0, i] 
			Yw = Pcal[1, i] 
			Zw = Pcal[2, i]
			A_line = np.array([[x, r_matrix[0,0]*Xw+r_matrix[0,1]*Yw+r_matrix[0,2]*Zw+Tx]])
			b_line = np.array([[-x*(r_matrix[2,0]*Xw+r_matrix[2,1]*Yw+r_matrix[2,2]*Zw)]])
			A = np.append(A, A_line, axis=0)
			b = np.append(b, b_line, axis=0)
		return self.solucao_axb(A, b)


	def calibracao(self, Pcal, Ical):

		lambda_abs, alpha, v_solucao = self.achar_lambda_alpha_v(Pcal, Ical)	

		lambda_abs_alpha = lambda_abs*alpha
		r_matrix = [[ v_solucao[4]/lambda_abs_alpha, v_solucao[5]/lambda_abs_alpha, v_solucao[6]/lambda_abs_alpha],
					[v_solucao[0]/lambda_abs, v_solucao[1]/lambda_abs, v_solucao[2]/lambda_abs]]
		r_third_row = np.cross(r_matrix[0], r_matrix[1])
		r_matrix = np.vstack((r_matrix, r_third_row))

		print(r_matrix)

		print(self.achar_tz_fx(Pcal, Ical, r_matrix))
	'''
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

	#	Pcal = np.append(Pcal, [np.ones(len(Pcal[0]))], axis=0)
#		Ical = np.append(Ical, [np.ones(len(Ical[0]))], axis=0)
#		m = np.matmul(Ical, np.linalg.pinv(Pcal))
#		m = np.reshape(m, (12))


		print('m', m)
		gamma_abs = np.sqrt(m[8]**2+ m[9]**2+ m[10]**2)
	
		print(gamma_abs)

		for i in range(12):
			m[i] = m[i]/gamma_abs

		print('m n', m)
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


		self.printMatrix(r_matrix)
		T = [ (m[3]-ox*m[11])/(-fx), (m[7]-oy*m[11])/(-fy), m[11]]

		print(T)