from copy import copy
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


