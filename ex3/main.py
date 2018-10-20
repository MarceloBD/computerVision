from projecao import Projecao


if __name__ == '__main__':

	proj = Projecao()

	pjp = proj.pontos_calibracao(3,3,3)

	proj.printMatrix(pjp)

	Ical = [16, 0.01, 0.01, 320, 240]

	proj.calibracao(pjp, Ical)