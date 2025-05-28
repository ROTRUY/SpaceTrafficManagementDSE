### IMPORTS
import matplotlib.pyplot as plt
from datetime import datetime

### FUNCTIONS

def read_gsc(file: str) -> list[list[datetime], list[datetime], list[float]]:
    """
    Function to read GMAT ground station contact time files into list of [[Start times], [End times], [Durations]].

    Inputs
    -----
    - `file:` File path of the file to be read. Format of lines in the data must be:
    01 Jan 2030 09:13:43.103    01 Jan 2030 09:19:04.115      321.01231974

    Outputs
    -----
    - `GSCD:` List of start times, end times and durations of ground station contacts in the following format:
    [[Start times], [End times], [Durations]]
    """
    GSCD: list[list[datetime], list[datetime], list[float]] = [[], [], []]  # [[Start times], [End times], [Durations]]
    
    with open(file) as f:  # Open the file
        for line in f.readlines()[4:-3]:  # Go through the lines, except the header lines and footer lines
            start = datetime.strptime(line[:20], "%d %b %Y %H:%M:%S")  # Get the start time
            end = datetime.strptime(line[28:48], "%d %b %Y %H:%M:%S")  # Get the end time
            duration = float(line[58:70])  # Get the duration

            # Append to GSCD to save
            GSCD[0].append(start)
            GSCD[1].append(end)
            GSCD[2].append(duration)
    
    return GSCD

def read_file(file):
    with open(file) as f:
        lines = f.readlines()
    
    for line in lines:
        if "Number of events" in line:
            # Extract the number after the colon and strip whitespace
            number = int(line.split(":")[-1].strip())
            return number

    return None  # If the line is not found

number_of_events50 = read_file('data/GSContactData50.txt')
number_of_events55 = read_file('data/GSContactData55.txt')
number_of_events58 = read_file('data/GSContactData58.txt')
number_of_events60 = read_file('data/GSContactData60.txt')
number_of_events65 = read_file('data/GSContactData65.txt')

print(f'Number of events for 50 degrees: {number_of_events50}') 
print(f'Number of events for 55 degrees: {number_of_events55}')
print(f'Number of events for 58 degrees: {number_of_events58}')
print(f'Number of events for 60 degrees: {number_of_events60}')
print(f'Number of events for 65 degrees: {number_of_events65}')

### MAIN
# Read data, save into list of lists
GSCD50 = read_gsc('data/GSContactData50.txt')
GSCD55 = read_gsc('data/GSContactData55.txt')
GSCD58 = read_gsc('data/GSContactData58.txt')
GSCD60 = read_gsc('data/GSContactData60.txt')
GSCD65 = read_gsc('data/GSContactData65.txt')



# Determine total duration 
total_duration50 = sum(GSCD50[2])
total_duration55 = sum(GSCD55[2])
total_duration58 = sum(GSCD58[2])
total_duration60 = sum(GSCD60[2])
total_duration65 = sum(GSCD65[2])


# Print total durations
print(f'Total duration for 50 degrees: {total_duration50} seconds')
print(f'Total duration for 55 degrees: {total_duration55} seconds')
print(f'Total duration for 60 degrees: {total_duration60} seconds')
print(f'Total duration for 65 degrees: {total_duration65} seconds')




no_contact_lst = []
# Get time between contacts
#for i in range(len(GSCD[0])-1):
#    no_contact = GSCD[0][i+1] - GSCD[1][i]
#    no_contact_lst.append(no_contact)

#print(max(no_contact_lst))

### PLOTTERDEPLOT
plot = False
nr = 50  # Number of durations to plot

if plot:
    x = range(len(GSCD[2]))
    if nr > len(x):  # Remove user errors
        nr = len(x)

    plt.bar(x[:nr], GSCD[2][:nr])
    plt.show()
