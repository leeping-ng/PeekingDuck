import logging
from models.peoplecounting.divider import DividerZone
from models.peoplecounting.area import Area


class PeopleCountingModel:
    """A class to do rule base inference for activity counting
    """

    def __init__(self, configs):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Creating people counting model.')
        self.name = 'peoplecounting'
        self.model_type = 'heuristics'

        discount_rate = configs['models']['peoplecounting']['discount_rate']
        self.discount_rate = float(discount_rate)

        self.conditions = configs['models']['peoplecounting']['conditions']
        if self.conditions:
            self.zones = []
            zones_info = configs['models']['peoplecounting']['zones']
            for zone in zones_info:
                # each zone is an array where [0] is the zone type (dividers or area)
                # then the relevant information for each zone type
                if zone[0] == "dividers":
                    zone = DividerZone(zone[1])
                    self.zones.append(zone)
                elif zone[0] == "area":
                    zone = Area(zone[1])
                    self.zones.append(zone)
                # is neither, something is wrong. Raise error
                else:
                    raise TypeError('Zone Type Error: {}.'.format(zone[0]))

        self.logger.info('discount rate: {:.2f}'.format(self.discount_rate))
        self.people_ids = []
        self.people_counter = 0

    def predict(self, image_size, bboxes, tracked=False):
        '''detect people using people tracker and count the number of people
        that has come into the stream/video. Returns count.'''

        # tracked_bbox[4] is unique id, tracked_bboxes[5] is class_index where 0 is person.
        # for every bboxes tracked by deepsort, only add to counter if detected person is a new id.
        # If new, add to counter and add id to list of people_ids to check against
        if tracked:
            for bbox_info in bboxes:
                # if (class is person) and (unique id) not in (already detected persons list)
                if (bbox_info[5] == 0) and (bbox_info[4] not in self.people_ids):
                    self.people_ids.append(bbox_info[4])
                    self.people_counter += 1
            return int(self.people_counter*self.discount_rate)

        # if not tracked and zones are set, use Dividers to determine if person is within the zone.
        # divider checks the xy point to see if it is within. Pass (x1-x2)/2 and y2
        # need to reset counter for every predict
        if self.conditions:
            num_of_zones = len(self.zones)
            zones_counts = [0]*num_of_zones
            for bbox in bboxes:
                x = ((bbox[0] + bbox[2]) / 2) * image_size[0]
                y = bbox[3] * image_size[1]

                # for each bbox, check if it is in any zone and add count
                for i, zone in enumerate(self.zones):
                    if zone.point_within_zone(x, y):
                        zones_counts[i] += 1

            return zones_counts

        # no tracking and no conditions, return total count of human bbox
        return len(bboxes)

    def train(self, input_dir, output_dir, use_case, epochs=1):
        """Hard coded rules don't need training"""
