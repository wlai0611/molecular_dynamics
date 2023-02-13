import time
def write_trajectory(trajectory):
    outstring = ""
    timestamp = str(round(time.time()))
    filename  = f"trajectory{timestamp}.xyz"
    connection= open(file=filename, mode='w')
    for positions in trajectory:
        m,n       = positions.shape
        outstring+=str(m)+"\n\n"
        for coordinates in positions:
            outstring+="LJ "+" ".join([str(coordinate) for coordinate in coordinates])+"\n"
    connection.write(outstring)