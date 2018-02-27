import sys

def parse_data(filename, seperator=',', data_format=None, newline=False):
	'''Parses data from a file specified
		Parameters:
			filename---- string, required. 
						 A file where the data to be parsed is stored
			seperator---- string, optional (default=",")
						  What defines the seperation between data attributes
			data_format---- list, optional (default=None)
						    A list of data types in the file, if they are known.
						    If param is specified by user, parser will attempt to cast
						    to data type specified for each line attribute. If a cast cannot
						    be completed, attribute is stored as a string
						    e.g., [float, float, float, float, str] for Iris dataset
			newline---- Bool, optional (default=False)
						If false and data contains newline chars, they are removed
		Returns:
			a list of lists containing parsed data from the file
	'''

	# Open the data file
	try:
		file = open(filename, 'r')
	except:
		print("Cannot open file \'"+str(filename)+"\'. Exiting.")
		sys.exit()

	data = []
	linenum = 0
	for line in file:
		
		data.append([])		# Start with empty line
		attr = ""			# Attribute to be read from data line
		
		for char in line:
			if char == seperator:
				if data_format is not None:
					# Try casting to data type specified in format. On failure, use string
					try:
						data[linenum].append(data_format[len(data[linenum])](attr))
					except:
						data[linenum].append(attr)
				else: 
					data[linenum].append(attr)

				attr = "" 	# Clear attribute to move to next one
			else: attr += char

		# Remove any new line characters from the end
		if not newline and attr[len(attr) - 1] == '\n': attr = attr[:len(attr) - 1]
		data[linenum].append(attr)
		linenum += 1

	file.close()

	return data

