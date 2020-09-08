import struct
num = 0
while 1:
	#fileName = 'data/smoke_gun/d/'+str(num).zfill(3)+'.npz'
	inFileName = str(num).zfill(3)+'.npz'
	outFileName = str(num).zfill(3)+'.vol'
	try:
		fin = open(inFileName,'rb')
	except Exception:
		print('all finished')
		break
	print('processing '+inFileName)
	try:
		fout = open(outFileName,'wb')
	except Exception:
		print("write file error")
		break
	fin.seek(3,0)
	print(struct.unpack('iiiiiiiiii',fin.read(40)))
	print(struct.unpack('i',fin.read(4)))
	fout.close()
	fin.close()
	num += 1
