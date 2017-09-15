class Node:

	def __init__(self):
		self.__index = -1
		#self.__net_value = -1
		return None
	
	#---------------------
	
	def get_index(self):
		return self.__index
	
	def set_index(self, index):
		self.__index = index
		
	def get_level(self):
		return self.__level
	
	def set_level(self, level):
		self.__level = level
	
	def get_item(self):
		return self.__item
	
	def set_item(self, item):
		self.__item = item
		
	def get_label(self):
		return self.__label
	
	def set_label(self, label):
		self.__label = label
	
	def get_layer_name(self):
		return self.__layer_name
	
	def set_layer_name(self, layer_name):
		self.__layer_name = layer_name
		
	def get_net_input_value(self):
		return self.__net_input_value
	
	def set_net_input_value(self, net_input_value):
		self.__net_input_value = net_input_value
	
	def get_net_value(self):
		return self.__net_value
	
	def set_net_value(self, net_value):
		self.__net_value = net_value
	
	def get_small_delta(self):
		return self.__small_delta
	
	def set_small_delta(self, small_delta):
		self.__small_delta = small_delta
	
	def get_is_bias_unit(self):
		return self.__is_bias_unit
	
	def set_is_bias_unit(self, is_bias_unit):
		self.__is_bias_unit = is_bias_unit	