

def get_global_id(id, type):
    '''
    The different entities may have same id (e.g. both author and conference can have id 1),
    to get a unique id for all entities in HIN (call it global id), we shift the origin id left by 2 bit,
    and use the lowest 2 bits to store the entity type information:

    global_id: xxxxx00 => Entity Type: author
    global_id: xxxxx01 => Entity Type: conference
    global_id: xxxxx10 => Entity Type: paper

    And we can retrieve the type information from global id by global_id & 3:
    global_id & 3 == 0 => Entity Type: author
    global_id & 3 == 1 => Entity Type: conference
    global_id & 3 == 2 => Entity Type: paper
    '''
    return (int(id) << 2) + type

def retrieve_type(global_id):
    return global_id & 3

def retrieve_id(global_id):
    return global_id >> 2