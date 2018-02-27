import sys

def parse_data(filename, seperator=','):
	
	# Open the data file
	try:
		file = open(filename, 'r')
	except:
		print("Cannot open file \'"+str(filename)+"\'. Exiting.")
		sys.exit()

	data = []
	linenum = 0
	for line in file:
		data.append([])
		attr = ""	# Attribute to be read from data line
		for char in line:
			if char == seperator: 
				data[linenum].append(attr)
				attr = ""
			else: attr += char

		# Remove any new line characters from the end
		if attr[len(attr) - 1] == '\n': attr = attr[:len(attr) - 1]
		data[linenum].append(attr)
		linenum += 1

	file.close()


parse_data("Iris.txt")

