from copy import copy
import math 
import numpy as np


class Projecao:

	##########################################################
	def proj_perspectiva_mm(self, P, f):
	# @P matriz 3xN em [mm]
	# @f distancia focal da camera [mm]
	# retorna projecao do ponto P [mm] 
		p = copy(P)
		for i in xrange(len(P[0])):
			p[0][i] = f*P[0][i]/P[2][i]
			p[1][i] = f*P[1][i]/P[2][i]
			p[2][i] = f
		return p

	def proj_perspectiva_pixel(self, p, sx, sy, ox, oy):
	# @p coordenadas a projecao do pontos [mm]
	# @sx dimensao horizontal do pixel [mm/pixel]
	# @sy dimensao vertical do pixel [mm/pixel]
	# @ox coordenada de projecao do eixo optico [pixel]
	# retorna coordenada do pixel [pixel]
		pixel = np.zeros(shape=(2, len(p[0])))
		for i in xrange(len(p[0])):
			pixel[0][i] = -int(p[0][i])/sx + ox 
			pixel[1][i] = -int(p[1][i])/sy + oy
		return pixel

	def printMatrix(self, matrix):
		for i in xrange(len(matrix)):
			for j in xrange(len(matrix[0])):
				print matrix[i][j],
			print			

	###########################################################


	def mundo_para_camera(Pw,H):
	# Pw matriz 3xN [mm] coordenada global
	# H matriz de transformacao homogenea [mm]
	# retorna ponto em coordenada da camera 3xN[mm]
	P = copy(Pw)
	Pc = copy(Pw)
	P = vstack(Pw,[1 for _ in range(Pw[0])])
		for n in range(len(P[0])):
			x,y,z = [0,0,0]
			for i in range(4):
				Pc[0][n] += H[0][i]*P[i][n]
				Pc[1][n] += H[1][i]*P[i][n]
				Pc[2][n] += H[2][i]*P[i][n]
		return Pc

	def homogenea(rotx, roty, rotz, dx, dy, dz):
	# rotx, roty, rotz rotacao em graus 
	# dx, dy, dz deslocamento em [mm]
	# retorna matriz de transformacao homogenea (3xN)
		rotxR = rotx*pi/180.0  
		rotyR = roty*pi/180.0 
		rotzR = rotz*pi/180.0  
		cz = cos(rotzR) 
		sz = sin(rotzR)
		cy = cos(rotyR) 
		sy = sin(rotyR)
		cx = cos(rotxR) 
		sx = sin(rotxR)

		H = [[cz*cy, -sz*cx+cz*sy*sx, sz*sx+cz*sy*cx,  -dx],
			 [sz*cy, cz*cx+sz*sy*sx,  -cz*sx+sz*sy*cx, -dy],
			 [-sy,   cy*sx,           cx*cy,           -dy]]
		return H