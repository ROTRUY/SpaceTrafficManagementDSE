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

### MAIN
# Read data, save into list of lists
GSCD = read_gsc('data/GSContactData55.txt')

# Determine total duration 
total_duration = sum(GSCD[2])

# Optionally print it in seconds, minutes, or as a timedelta
print(f"Total Duration: {total_duration:.2f} seconds")

# In minutes
print(f"Total Duration: {total_duration / 60:.2f} minutes")

no_contact_lst = []
# Get time between contacts
for i in range(len(GSCD[0])-1):
    no_contact = GSCD[0][i+1] - GSCD[1][i]
    no_contact_lst.append(no_contact)

print(max(no_contact_lst))

### PLOTTERDEPLOT
plot = False
nr = 50  # Number of durations to plot

if plot:
    x = range(len(GSCD[2]))
    if nr > len(x):  # Remove user errors
        nr = len(x)

    plt.bar(x[:nr], GSCD[2][:nr])
    plt.show()
