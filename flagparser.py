import json
import inflect

inflect = inflect.engine()

class FlagParser:
	def __init__(self, cmd):
		args = cmd.split()
		self.flags = {}
		current_key = False

		for index in range(len(args)):       
				if '--' in args[index]:
					current_key = args[index][2:]
					self.flags[current_key] = []
				elif current_key is not False:
					self.flags[current_key].append(args[index])

		for key in self.flags.keys():
			if inflect.singular_noun(key) == False and len(self.flags[key]) == 1:
				if self.flags[key][0].isnumeric():
					self.flags[key] = int(self.flags[key][0])
				else:
					self.flags[key] = self.flags[key][0]

	def __repr__(self):
		return json.dumps(self.flags)

	def get(self, key):
		value = False
		try:
			value = self.flags[key]
		except:
			value = False

		return value