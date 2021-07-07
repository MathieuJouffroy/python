import sys
import string
import time

def text_analyser(text=""):	
	count_upper = 0
	count_lower = 0
	count_space = 0
	count_marks = 0
	if not text:
		text_analyser(input ("What is the text to analyse?\n"))
	else:
		for letter in text:
			if letter.isupper():
				count_upper += 1
			if letter.islower():
				count_lower += 1
			if letter.isspace():
				count_space += 1
			if letter in string.punctuation:
				count_marks += 1
		print ("The text contains {} characters:"\
		.format(count_marks + count_upper + count_lower + count_space))
		print ("- {0} upper letters\n- {1} lower letters\n- {2} spaces\n- {3} punctuation marks"\
		.format(count_upper, count_lower, count_space, count_marks))


def text_analyser2(text=""):
	upper = len([letter for letter in text if letter.isupper()])
	lower = len([letter for letter in text if letter.islower()])
	space = len([letter for letter in text if letter.isspace()])
	marks = len([letter for letter in text if letter in string.punctuation])
	print (f"The text contains {upper + lower + space + marks} characters:")
	print (f"- {upper} upper letters\n- {lower} lower letters\n- {space} spaces\n- {marks} punctuation marks")

argv_len = len (sys.argv) - 1
if argv_len != 1:
	text_analyser()
else:
	start = time.time()
	text_analyser(sys.argv[1])
	end = time.time()
	print(f"text_analyser:\t[ exec-time = {end - start:.7f} ms ]")
	start = time.time()
	text_analyser2(sys.argv[1])
	end = time.time()
	print(f"text_analyse2r2:\t[ exec-time = {end - start:.7f} ms ]")
