from copy import copy
import numpy as np


class Projecao:

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
