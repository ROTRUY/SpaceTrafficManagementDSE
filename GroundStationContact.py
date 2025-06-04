### IMPORTS
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

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

number_of_events42 = read_file('data/GSContactData42.txt')
number_of_events45 = read_file('data/GSContactData45.txt')
number_of_events50 = read_file('data/GSContactData50.txt')
number_of_events55 = read_file('data/GSContactData55.txt')
number_of_events58 = read_file('data/GSContactData58.txt')
number_of_events59 = read_file('data/GSContactData59.txt')
number_of_events60 = read_file('data/GSContactData60.txt')
number_of_events61 = read_file('data/GSContactData61.txt')
number_of_events62 = read_file('data/GSContactData62.txt')
number_of_events63 = read_file('data/GSContactData63.txt')
number_of_events64 = read_file('data/GSContactData64.txt')
number_of_events65 = read_file('data/GSContactData65.txt')
number_of_events70 = read_file('data/GSContactData50.txt')
number_of_events80 = read_file('data/GSContactData50.txt')
number_of_events98 = read_file('data/GSContactData98.txt')
inclinations = [42, 45, 50, 55, 58,59, 60, 61, 62, 63, 64, 65, 70, 80, 98]  # Match the number of total_durations entries
# Print number of events table
number_of_events = [
    number_of_events42,
    number_of_events45,
    number_of_events50,
    number_of_events55,
    number_of_events58,
    number_of_events59, 
    number_of_events60,
    number_of_events61,
    number_of_events62,
    number_of_events63,
    number_of_events64,
    number_of_events65,
    number_of_events70,
    number_of_events80,
    number_of_events98
]

print("\nInclination (deg) | Number of Events")
print("------------------|-----------------")
for inc, num in zip(inclinations, number_of_events):
    print(f"{inc:<17} | {num}")
# Print number of events    

### MAIN
# Read data, save into list of lists
GSCD42 = read_gsc('data/GSContactData42.txt')
GSCD45 = read_gsc('data/GSContactData45.txt')
GSCD50 = read_gsc('data/GSContactData50.txt')
GSCD55 = read_gsc('data/GSContactData55.txt')
GSCD58 = read_gsc('data/GSContactData58.txt')
GSCD59 = read_gsc('data/GSContactData59.txt')
GSCD60 = read_gsc('data/GSContactData60.txt')
GSCD61 = read_gsc('data/GSContactData61.txt')
GSCD62 = read_gsc('data/GSContactData62.txt')
GSCD63 = read_gsc('data/GSContactData63.txt')
GSCD64 = read_gsc('data/GSContactData64.txt')  
GSCD65 = read_gsc('data/GSContactData65.txt')
GSCD70 = read_gsc('data/GSContactData70.txt')
GSCD80 = read_gsc('data/GSContactData80.txt')
GSCD98 = read_gsc('data/GSContactData98.txt')
GSCDyear = read_gsc('data/GroundStationContactData.txt')
GSCDMatera = read_gsc('data/GSContactData60Matera.txt')
GSCDPotsdam = read_gsc('data/GSContactData60Potsdam.txt')


# Plot inclinations vs number of GSCD events
gscd_lengths = [
    len(GSCD42[0]),
    len(GSCD45[0]),
    len(GSCD50[0]),
    len(GSCD55[0]),
    len(GSCD58[0]),
    len(GSCD59[0]),
    len(GSCD60[0]),
    len(GSCD61[0]),
    len(GSCD62[0]),
    len(GSCD63[0]),
    len(GSCD64[0]),
    len(GSCD65[0]),
    len(GSCD70[0]),
    len(GSCD80[0]),
    len(GSCD98[0])
]

plt.figure()
plt.plot(inclinations, gscd_lengths, marker='o')
plt.xlabel('Inclination (deg)')
plt.ylabel('Visition Frequency (Number of Contacts)')
#plt.title('Visition Frequency vs Inclination')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/VisitFreqIfoInc")
# Print the number of contacts

# Determine total duration
total_duration42 = sum(GSCD42[2])
total_duration45 = sum(GSCD45[2])
total_duration50 = sum(GSCD50[2])
total_duration55 = sum(GSCD55[2])
total_duration58 = sum(GSCD58[2])
total_duration59 = sum(GSCD59[2])
total_duration60 = sum(GSCD60[2])
total_duration61 = sum(GSCD61[2])
total_duration62 = sum(GSCD62[2])
total_duration63 = sum(GSCD63[2])
total_duration64 = sum(GSCD64[2])
total_duration65 = sum(GSCD65[2])
total_duration70 = sum(GSCD70[2])
total_duration80 = sum(GSCD80[2])
total_duration98 = sum(GSCD98[2])
total_durationyear = sum(GSCDyear[2])
print(total_durationyear)

# Convert total durations from seconds to hours
total_durations = [
    total_duration42 / 3600,
    total_duration45 / 3600,
    total_duration50 / 3600,
    total_duration55 / 3600,
    total_duration58 / 3600,
    total_duration59 / 3600,
    total_duration60 / 3600,
    total_duration61 / 3600,
    total_duration62 / 3600,
    total_duration63 / 3600,
    total_duration64 / 3600,
    total_duration65 / 3600,
    total_duration70 / 3600,
    total_duration80 / 3600,
    total_duration98 / 3600
]

# Plot inclinations vs total duration (hours)
plt.figure()
plt.plot(inclinations, total_durations, marker='o')
plt.xlabel('Inclination (deg)')
plt.ylabel('Total Duration (hours)')
#plt.title('Total Contact Duration vs Inclination')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/TotDurIfoInc")  

# Create a table of inclination and total duration

total_durations = [
    total_duration42,
    total_duration45,
    total_duration50,
    total_duration55,
    total_duration58,
    total_duration59,
    total_duration60,
    total_duration61,
    total_duration62,
    total_duration63,
    total_duration64,
    total_duration65,
    total_duration70,
    total_duration80,
    total_duration98
]

print("\nInclination (deg) | Total Duration (s)")
print("------------------|-------------------")
for inc, dur in zip(inclinations, total_durations):
    print(f"{inc:<17} | {dur:.2f}")




no_contact_lst = []
# Get time between contacts
#for i in range(len(GSCD[0])-1):
#    no_contact = GSCD[0][i+1] - GSCD[1][i]
#    no_contact_lst.append(no_contact)

#print(max(no_contact_lst))

### PLOTTERDEPLOT
# plot = False
# nr = 50  # Number of durations to plot

# if plot:
#     x = range(len(GSCD[2]))
#     if nr > len(x):  # Remove user errors
#         nr = len(x)

#     plt.bar(x[:nr], GSCD[2][:nr])
#     plt.show()

def find_day_with_highest_total_contact(GSCD):
    """
    Finds the day with the highest total contact time (sum of durations) from the GSCD data.
    Returns a tuple: (date, total_duration_in_seconds)
    """
    if not GSCD[0] or not GSCD[2]:
        return None, None

    day_totals = defaultdict(float)
    for start_time, duration in zip(GSCD[0], GSCD[2]):
        day = start_time.date()
        day_totals[day] += duration

    max_day = max(day_totals, key=day_totals.get)
    return max_day, day_totals[max_day]

def find_day_with_lowest_total_contact(GSCD):
    """
    Finds the day with the lowest total contact time (sum of durations) from the GSCD data.
    Returns a tuple: (date, total_duration_in_seconds)
    """
    if not GSCD[0] or not GSCD[2]:
        return None, None

    day_totals = defaultdict(float)
    for start_time, duration in zip(GSCD[0], GSCD[2]):
        day = start_time.date()
        day_totals[day] += duration

    min_day = min(day_totals, key=day_totals.get)
    return min_day, day_totals[min_day]

# Example usage:
highest_day, highest_total = find_day_with_highest_total_contact(GSCD60)
print(f"Day with highest total contact: {highest_day}, Total duration (s): {highest_total:.2f}")

lowest_day, lowest_total = find_day_with_lowest_total_contact(GSCD60)
print(f"Day with lowest total contact: {lowest_day}, Total duration (s): {lowest_total:.2f}")

def count_and_exclude_short_contacts(GSCD, station_name=""):
    """
    Counts and excludes contacts that are 30 seconds or less in duration from the GSCD data.
    Prints the number of passes >= 30 seconds and states if any were removed.
    Returns the filtered GSCD list.
    """
    filtered_GSCD = [[], [], []]
    removed_count = 0

    for start, end, duration in zip(GSCD[0], GSCD[1], GSCD[2]):
        if duration > 30:
            filtered_GSCD[0].append(start)
            filtered_GSCD[1].append(end)
            filtered_GSCD[2].append(duration)
        else:
            removed_count += 1

    kept_count = len(filtered_GSCD[0])
    print(f"{station_name}: Passes >= 30s: {kept_count}")
    if removed_count > 0:
        print(f"{station_name}: {removed_count} passes were removed (<= 30s).")
    else:
        print(f"{station_name}: No passes were removed.")

    return filtered_GSCD

# Example usage for Matera and Potsdam:
filtered_Matera = count_and_exclude_short_contacts(GSCDMatera, "Matera")
filtered_Potsdam = count_and_exclude_short_contacts(GSCDPotsdam, "Potsdam")
print(f"Number of passes for Matera (>=30s): {len(filtered_Matera[0])}")
print(f"Number of passes for Potsdam (>=30s): {len(filtered_Potsdam[0])}")

def plot_filtered_contacts(GSCD, station_name):
    """
    Plots a graph with dates on the x-axis and contact durations (in seconds) on the y-axis for the given GSCD data.
    Adds a red horizontal line at 30 seconds.
    """
    import matplotlib.dates as mdates

    if not GSCD[0]:
        print(f"No data to plot for {station_name}.")
        return

    dates = [dt.date() for dt in GSCD[0]]
    durations = GSCD[2]

    plt.figure(figsize=(10, 5))
    plt.scatter(dates, durations, s=10, label='pass')
    plt.axhline(30, color='red', linestyle='--', label='30 seconds')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlabel('Date')
    plt.ylabel('Contact Duration (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(loc='lower left')  # Move legend to upper left
    plt.grid(True)
    plt.show()

# Plot for Matera and Potsdam
plot_filtered_contacts(filtered_Matera, "Matera")
plot_filtered_contacts(filtered_Potsdam, "Potsdam")
