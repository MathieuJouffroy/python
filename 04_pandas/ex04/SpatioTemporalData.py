from FileLoader import FileLoader

class SpatioTemporalData():
	
	def __init__(self, df):
		self.df = df
	def when(self, location) -> list:
		return list(self.df[self.df['City'] == location]['Year'].drop_duplicates())
	def where(self, date) -> list:
		return list(self.df[self.df['Year'] == date]['City'].drop_duplicates())

loader = FileLoader()
data = loader.load("../resources/athlete_events.csv")
sp = SpatioTemporalData(data)
print (sp.when('Athina'))
print (sp.when('Paris'))
print (sp.where(1992))