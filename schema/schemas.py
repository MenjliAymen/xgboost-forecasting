def individual_serial(Chart)->dict:
    return {
        "id":str(Chart["_id"]),
        "time":Chart["time"],
        "value":Chart["value"]
    }
def list_serial(Charts)->list:
    return [individual_serial(Chart) for Chart in Charts]