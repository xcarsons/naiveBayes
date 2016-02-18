# Patient object
class Patient(object):
	id = 0
	gender = ""
	virus = None
	blood = ""
	weight = None

	# Patient constructor
	def __init__(self, id, gender, blood, weight, virus):
		super(Patient, self).__init__()
		self.id = id
		self.gender = gender
		self.virus = virus
		self.blood = blood
		self.weight = weight
		