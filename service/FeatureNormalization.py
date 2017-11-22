class FeatureNormalization:
	
	def normalize(instances, activation_function):
		
		#find minimum and maximum values for features
		
		num_of_features = len(instances[0])
		
		max_items = []
		min_items = []
		
		for j in range(num_of_features): #columns
			
			temp_max = instances[0][j]
			temp_min = instances[0][j]
			
			for i in range(len(instances)): #rows
				
				instance = instances[i][j]
				
				if instance > temp_max:
					temp_max = instance
				
				if instance < temp_min:
					temp_min = instance
					
			#row iteration end		
			max_items.append(temp_max)	
			min_items.append(temp_min)
			
			temp_max = instances[0][j]
			temp_min = instances[0][j]	
		
		#column iteration end	
		
		#-------------------------
		
		#normalization logic
		
		for i in range(len(instances)):
			
			for j in range (num_of_features):
				
				value = instances[i][j]
				
				maxItem = max_items[j]
				minItem = min_items[j]
				
				if j == num_of_features - 1: #output
									
					newMax = 1
					newMin = 0
				
				else: #input features
					
					newMax = +4
					newMin = -4
					
				value = ((newMax - newMin)*((value - minItem) / (maxItem - minItem))) + newMin
				
				instances[i][j] = value
			
		return instances
	