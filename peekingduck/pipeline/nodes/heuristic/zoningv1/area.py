from shapely.geometry.polygon import Polygon, Point
from models.peoplecounting.zone import Zone


class Area(Zone):
    """This is a zone subclass that uses polygon area to create a zone.
    """

    def __init__(self, coord_list):
        super().__init__("polygon")
        self.polygon_points = [tuple(x) for x in coord_list]
        self.polygon = Polygon(self.polygon_points)

    def point_within_zone(self, x, y):
        """Function used to check whether the bottom middle point of the bounding box
        is within the stipulated zone created by the divider.

        Args:
            x (float): middle x position of the bounding box
            y (float): lowest y position of the bounding box

        Returns:
            boolean: whether the point given is within the zone.
        """
        return self._is_inside(x, y)

    def get_all_points_of_area(self):
        """Function used to Get all (x, y) tuple points that form the area of the zone.

        Args:
            None

        Returns:
            list: returns a list of (x, y) points that form the zone area.
        """
        return self.polygon_points

    def _is_inside(self, x, y):
        point = Point((x, y))
        return self.polygon.contains(point)
