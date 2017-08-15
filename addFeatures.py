def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    
    all_messages = float(all_messages)
    poi_messages = float(poi_messages)
    
    
    import math
    if not (math.isnan(poi_messages) or math.isnan(all_messages)):
        fraction = poi_messages / all_messages

    return fraction



def poi_interaction_ratios(data_dict):
    """
    :param data_dict: List of employees and their features
    :return: Populates the `to_poi_ratio` and `from_poi_ratio` keys for
             each person in the dataset as either a number or `NaN` and returns
             a new data_dict
    """
    for name in data_dict:

        data_point = data_dict[name]

        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["from_poi_ratio"] = fraction_from_poi


        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["to_poi_ratio"] = fraction_to_poi
        
    return data_dict  