import os
import glob

def parse_solomon_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    name = lines[0].strip()
    vehicle_line = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('VEHICLE'):
            vehicle_line = idx
            break

    if vehicle_line is None:
        raise ValueError("VEHICLE line not found.")

    # Vehicle info
    cap_line = lines[vehicle_line + 2].split()
    vehicle_count = int(cap_line[0])
    vehicle_capacity = int(cap_line[1])

    # Find start of customer section
    cust_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('CUSTOMER'):
            cust_start = idx + 2  # skip header line
            break
    if cust_start is None:
        raise ValueError("CUSTOMER line not found.")

    customers = []
    for line in lines[cust_start:]:
        if not line.strip():
            continue
        parts = line.split()
        #Customer number, x, y, demand
        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = int(parts[3])
        customers.append((cust_id, x, y, demand))

    return {
        'name': name,
        'vehicle_count': vehicle_count,
        'vehicle_capacity': vehicle_capacity,
        'customers': customers
    }

def write_reduced(customers, outpath, instance_name, vehicle_count, capacity):
    with open(outpath, 'w') as f:
        f.write(f"NAME: {instance_name}\n")
        f.write(f"VEHICLES: {vehicle_count}\n")
        f.write(f"CAPACITY: {capacity}\n")
        f.write("CUSTNO XCOORD YCOORD DEMAND\n")
        for cust_id, x, y, demand in customers:
            f.write(f"{cust_id} {x} {y} {demand}\n")

def main():
    # Use glob to find all input files
    input_files = glob.glob("./solomon100_instances/*.txt")
    output_dir = "reduced_instances"
    os.makedirs(output_dir, exist_ok=True)

    for infile in input_files:
        data = parse_solomon_file(infile)
        customers = data['customers']
        name = os.path.splitext(os.path.basename(infile))[0]

        for n, tag in zip([25, 50, 100], ['25', '50', '100']):
            reduced_customers = customers[:n+1]  # +1 to include depot
            outname = f"{name}_{tag}.txt"
            outpath = os.path.join(output_dir, outname)
            write_reduced(
                reduced_customers, outpath,
                instance_name=name,
                vehicle_count=data['vehicle_count'],
                capacity=data['vehicle_capacity']
            )
            print(f"Written {outpath} ({len(reduced_customers)} customers)")

if __name__ == "__main__":
    main()
