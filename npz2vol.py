import struct
import numpy as np

prefix=""

def main():
	num = 0
	while npz2vol(prefix+str(num).zfill(3)):
		num += 1
	print("complete")

def npz2vol(fileName):
	inFileName = fileName+'.npz'
	outFileName = str(fileName).zfill(3)+'.vol'
	print('processing '+inFileName)
	try:
		fin = np.load(inFileName)
	except Exception:
		print('read file '+inFileName+' error')
		return False
	try:
		fout = open(outFileName,'wb')
	except Exception:
		print("write file "+outFileName+" error")
		return False
	d = fin['x']
	zRes = len(d)
	yRes = len(d[0])
	xRes = len(d[0][0])
	channel = 1
	#'V''O''L'
	fout.write("VOL".encode('UTF-8'))
	#version = 3
	fout.write(struct.pack('b',3))
	#encoding identifier
	fout.write(struct.pack('i',1))
	#resolution
	fout.write(struct.pack('iiii',xRes,yRes,zRes,channel))
	#aabb min max
	fout.write(struct.pack('ffffff',-0.5,-1.0,-0.5,0.5,1.0,0.5))
	#data
	fout.write(d.tobytes())

	fout.close()
	fin.close()
	return True

if __name__ == '__main__':
	main()