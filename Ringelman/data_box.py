class DataBox():
    def __init__(self,rectangles=None,polygons=None):
        self.rectangles=[]
        self.polygons = []
        if not rectangles is None: self.rectangles.extend(rectangles)
        if not polygons is None: self.polygons.extend(polygons)
        self.RECTANGLE = 0
        self.POLYGON = 1

    def add(self,points):
        if len(points)==2:
            self.rectangles.append(points)
        elif len(points) == 4:
            self.polygons.append(points)
        else:
            raise Exception("错误的数据")
    def extend(self,datas,mode):
        if mode == self.RECTANGLE:
            self.rectangles.extend(datas)
        elif mode == self.POLYGON:
            self.polygons.extend(datas)
        else:
            raise Exception("错误的类型数据")