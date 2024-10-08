import json
from ipaddress import IPv4Address, IPv4Network
import glob
import ipaddress
import pandas as pd
from datetime import datetime
import os
from internal_ip_range import internal_ip
from thefuzz import fuzz
from datetime import date
from dotenv import load_dotenv

load_dotenv()

working_directory = os.getenv('DATA_DIRECTORY')
cidr_location = os.getenv('CIDR_LOCATION')



def fetch_files(path, file_dict, file_id, file_list):
    for file in glob.glob(f"{path}/*"):
        if os.path.isdir(file):
            subdir = file
            fetch_files(subdir, file_dict, file_id, file_list)
        else:
            if os.path.isfile(file):
                if ".txt" in file:
                    file_list.append(file)
                    directory = path + "/"
                    filename = file.split(directory)[1]
                    if filename not in file_dict.keys():
                        file_dict[filename] = file_id
                        file_id += 1
    return file_dict, file_list

def fuzz_distance (a, b):
    distance = fuzz.ratio (a,b)/100
    return distance

def ip_location(ip, ip_network_list):
    Found = False
    for network in ip_network_list:
        if IPv4Address(ip) in IPv4Network(network):
            Found = True
            break
    return Found

def read_cidr(file):
    ip_network_list = []
    cidr_file = pd.read_csv(file)
    columns = cidr_file.shape[1]
    for i in range(columns-1):
        ip_network_list += cidr_file.iloc[:,i+1].values.tolist()
        return ip_network_list



def find_ip(string, source_alias):
    call_details = json.loads(string)
    content = call_details["Call"]
    ce_lic = "0"
    cloud_lic = "0"
    ip = ""
    if content.get("Legs") != None:
       if isinstance(content["Legs"], list):
          for leg in content["Legs"]:
              if leg.get("Leg") != None:
                  if leg["Leg"].get("SIP") != None:
                     if leg["Leg"]["SIP"].get("Address") != None:
                        if leg["Leg"]["SIP"] != "Not set" and leg["Leg"]["SIP"]["Address"] != "":
                            if leg["Leg"]["SIP"].get("Aliases") != None:
                               for item in leg["Leg"]["SIP"]["Aliases"]:
                                   if item["Alias"]["Value"] == source_alias:
                                       ip = leg["Leg"]["SIP"]["Address"].split(":")[0]
                                       break
    if content.get("License") != None:
       ce_lic = content["License"]["CollaborationEdge"]
       cloud_lic = content["License"]["Cloud"]

    return ip, ce_lic, cloud_lic
today = date.today()

#read network list
ip_network_list = read_cidr(cidr_location)
internal_ip_v4 = [ipv4 for ipv4 in internal_ip if ":"  not in ipv4]
internal_ip = ip_network_list + internal_ip_v4


max_ips = 0
data_list = []


file_dict, file_list = fetch_files(working_directory, {}, 0, [])

for file in file_list: #glob.glob(f"{directory}/*.txt"):
    print("analyzing ", file)
    with open(file, 'r', encoding='utf-8') as local_file:
        raw_data = json.load(local_file)
    for call in raw_data:
        if "H323" not in call["protocol"]:
            item = []
            source_alias = call["source_alias"]
            destination_alias = call["destination_alias"]
            licensed = call["licensed"]
            traversal = call["licensed_as_traversal"]
            #distance = fuzz_distance(source_alias, destination_alias)
            start_time = call["start_time"]
            fmt = '%Y-%m-%d %H:%M:%S.%f'
            moment = datetime.strptime(start_time, fmt)
            seconds = moment.timestamp()
            start_day = moment.strftime("%d/%m/%Y")
            start_hour = moment.strftime("%H:%M:%S")
            end_time = call["end_time"]
            disconnect_reason = call["disconnect_reason"]
            disconnect_key = disconnect_reason.split(" ")[0]
            if disconnect_key.lower() == "cancel" or disconnect_key.lower() == "bye":
               disconnect_key = '487'
            if disconnect_key == "disconnected":
               disconnect_key = "200"
            disconnect_key = int(disconnect_key)
            duration = datetime.strptime(end_time, fmt) - datetime.strptime(start_time, fmt)
            duration_sec = round(duration.total_seconds())
            details = call["details"]
            location = "Not set"
            ip, ce_lic, cloud_lic = find_ip(details, source_alias)
            source_alias = source_alias.split("sip:")[1]
            if "@" in source_alias:
                source_alias_user = source_alias.split("@")[0]
                source_alias_domain = source_alias.split("@")[1]
            else:
                source_alias_user = source_alias
                source_alias_domain = source_alias
            destination_alias = destination_alias.split("sip:")[1]
            if "@" in destination_alias:
                destination_alias_user = destination_alias.split("@")[0]
                destination_alias_domain = destination_alias.split("@")[1]
            else:
                destination_alias_user = destination_alias
                destination_alias_domain = destination_alias
            distance_user = fuzz_distance(source_alias_user, destination_alias_user)
            distance_domain = fuzz_distance(source_alias_domain, destination_alias_domain)
            if ip != "":
                ip_integer = int(ipaddress.ip_address(ip))
                Found = ip_location(ip, internal_ip)
                if Found == True:
                   location = "Internal"
                else:
                    location = "External"
                item = [source_alias, destination_alias, source_alias_user, source_alias_domain, destination_alias_user, destination_alias_domain, distance_user, distance_domain, disconnect_reason, disconnect_key, seconds, start_day, start_hour, duration_sec, ip, ip_integer, licensed, traversal, ce_lic, cloud_lic, location, file.split("/")[-1]]
                data_list.append(item)


columns = ["Source Alias", "Destination Alias", "Source User Alias", "Source Domain Alias", "Destination User Alias", "Destination Domain Alias", "User Distance", "Domain Distance", "Disconnect Reason", "Disconnect Key", "Day and time (sec)", "Day", "Time", "Duration (sec)", "Source IP", "Source IP (integer)", "Licensed", "Traversal", "CollaborationEdge", "Cloud", "Source IP Location", "Original File"]
data = pd.DataFrame(data_list, columns=columns)
ip_list = data["Source IP"].values.tolist()

ip_list_no_zeroes = list(filter(None, ip_list))
ip_list_no_duplicates = list(dict.fromkeys(ip_list_no_zeroes))

data.to_csv("global.csv")
spam_calls_original_index = data.loc[((data["Disconnect Reason"] == "404 Not Found") | (data["Disconnect Reason"] == "403 Forbidden")) & (data["Source IP Location"] == "External") & (data["CollaborationEdge"] == "0") & (data["Cloud"] == "0")]
legitimate_calls_original_index = data.loc[(data["Disconnect Reason"] != "404 Not Found") & (data["Disconnect Reason"] != "403 Forbidden") & (data["Duration (sec)"] > 10)] #& (data["CollaborationEdge"] == "0")
all_other_calls_original_index = data.loc[(data["Disconnect Reason"] != "404 Not Found") & (data["Disconnect Reason"] != "403 Forbidden") & (data["Duration (sec)"] <= 10)] #& (data["CollaborationEdge"] == "0")
legitimate_calls = legitimate_calls_original_index.reset_index().drop(["index"], axis=1)
spam_calls = spam_calls_original_index.reset_index().drop(["index"], axis=1)
all_other_calls = all_other_calls_original_index.reset_index().drop(["index"], axis=1)
spam_ip = data["Source IP"].values.tolist()

spam_ip_no_zeroes = list(filter(None, ip_list))
spam_ip_no_duplicates = list(dict.fromkeys(spam_ip_no_zeroes))
print("number of spam ips: ", len(spam_ip_no_duplicates))
#add "Pass" column
legitimate_calls["Pass"] = "y"
spam_calls["Pass"] = "n"
all_other_calls["Pass"] = "y"

spam_calls.to_csv(f"{working_directory}/spam_{today}.csv")
legitimate_calls.to_csv(f"{working_directory}/legitimate_{today}.csv")
all_other_calls.to_csv(f"{working_directory}/all_others_{today}.csv")



all_calls_unordered = pd.concat([legitimate_calls, spam_calls, all_other_calls], ignore_index=True)
all_calls = all_calls_unordered.drop(["Unnamed: 0"], axis=1, errors='ignore').sort_values(by=["Day and time (sec)"]).reset_index(drop=True)
all_calls.to_csv(f"{working_directory}/all_calls_{today}.csv")


