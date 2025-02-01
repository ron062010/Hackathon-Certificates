import random
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
# Job and VM Initialization with Extended Parameters

df = pd.read_csv(f'C:\\Users\\HP\\Desktop\\MOCK_DATA_50.csv')

jobs = df.to_dict(orient='records')

vms = [
    {
        "VM_ID": f"VM1",
        "Type_of_Service": ["SAAS", "PAAS", "IAAS"],
        "Job_Type": ["Computation-heavy", "I-O heavy", "Memory-heavy"],
        "Million_instruction_per_second": 1000,
        "Cost_Per_Instruction": 0.02,
        "Utilization_Capacity": 0.8,
        "Energy_Consumed_Per_Instruction": 0.0001,
        "cpu_capacity":10,
        "memory_capacity": 300
    },
    {
        "VM_ID": f"VM2",
        "Type_of_Service": ["IAAS", "PAAS", "SAAS"],
        "Job_Type": ["Memory-heavy", "Computation-heavy", "I-O heavy"],
        "Million_instruction_per_second": 800,
        "Cost_Per_Instruction": 0.015,
        "Utilization_Capacity": 0.85,
        "Energy_Consumed_Per_Instruction": 0.00015,
        "cpu_capacity":15,
        "memory_capacity": 300
    },
    {
        "VM_ID": f"VM3",
        "Type_of_Service": ["SAAS", "IAAS", "PAAS"],
        "Job_Type": ["I-O heavy", "Memory-heavy", "Memory-heavy"],
        "Million_instruction_per_second": 500,
        "Cost_Per_Instruction": 0.01,
        "Utilization_Capacity": 0.9,
        "Energy_Consumed_Per_Instruction": 0.0002,
        "cpu_capacity":16,
        "memory_capacity": 300
    },
]

def genetic_algorithm(jobs, vms, pop_size, generations, mutation_rate, crossover_rate):
    num_jobs = len(jobs)
    num_vms = len(vms)
    best_fitness_over_time = []

    def initialize_population(pop_size, num_jobs, num_vms):
        return [
            [random.randint(0, num_vms - 1) for _ in range(num_jobs)]
            for _ in range(pop_size)
        ]

    def fitness_function(chromosome, jobs, vms):

        #Shows at what time the VM is available for the next job to take up
        vm_available_time = {vm["VM_ID"]: 0 for vm in vms}

        #Storing the utilization of each VM
        vm_utilization = {vm["VM_ID"]: 0 for vm in vms}

        #Initial vars of EACH chromosome
        total_cost = 0
        total_energy = 0
        penalty = 0
        failed_jobs = []
        
        #The kind of main list which will tell which all jobs fall under which VM
        vm_job_mapping = {vm["VM_ID"]: [] for vm in vms}
        vms = sorted(vms, key=lambda vm: vm["VM_ID"])  
        #The below for loop will assign each job a VM
        for job_index, vm_id in enumerate(chromosome):

            #Pick a job one by one
            job = jobs[job_index]

            #VM assigned by the chromosome var
            vm = vms[vm_id]
            
            #Append that job to the main list.
            vm_job_mapping[vm["VM_ID"]].append(job)

        #In below for loop, AFTER the VM is assigned to a job, we check their compatibility and 
        # calculate the fitness value of the chromosome.
        for vm_id, assigned_jobs in vm_job_mapping.items():
            assigned_jobs.sort(key=lambda job: job["Priority"], reverse=True)

            for job in assigned_jobs:
                number_of_penalties = 0

                #Define job vars
                arrival_time = job["Arrival_Time"]
                instruction_count = job["EIC"]
                deadline = job["Deadline"]
                delay_cost = job["Delay_Cost"]
                cpu_requested_by_job = job["CPU_Requested"]
                memory_requested_by_job = job["Memory_Requested"]

                vm = [v for v in vms if v["VM_ID"] == vm_id][0]

                # Validate job and VM compatibility
                if job["Type_of_Service"] not in vm["Type_of_Service"]:
                    penalty += 1
                    failed_jobs.append({**job, "Failure_Reason": "Type of Service"})
                    continue

                if job["Job_Type"] not in vm["Job_Type"]:
                    penalty += 1
                    failed_jobs.append({**job, "Failure_Reason": "Type of Job"})
                    continue

                if vm["cpu_capacity"] < job["CPU_Requested"]:
                    penalty += 0.5
                    failed_jobs.append({**job, "Failure_Reason": "CPU Capacity Exceeded"})
                    continue

                if vm["memory_capacity"] < job["Memory_Requested"]:
                    penalty += 0.4
                    failed_jobs.append({**job, "Failure_Reason": "Memory Capacity Exceeded"})
                    continue

                #Check Job-VM Utilization compatbility
                execution_time = instruction_count / vm["Million_instruction_per_second"]
                utilization_increment = execution_time / (60 * 60)
                if vm_utilization[vm_id] + utilization_increment > vm["Utilization_Capacity"]:
                    penalty += 0.4
                    failed_jobs.append({**job, "Failure_Reason": "Utilization Capacity Exceeded"})
                    continue

                #Check if the VM is available to execute the complete job as no pre-emption is considered.
                #In that case, if VM is unavailable, add penalty.
                start_time = max(arrival_time, vm_available_time[vm["VM_ID"]])
                finish_time = start_time + execution_time
                if finish_time > deadline:
                    penalty += 0.5
                    failed_jobs.append({**job, "Failure_Reason": "Deadline Missed"})
                    continue

                #If the code reaches here, this means that there is 100% compatibility between VM-Job
                #Update the required parameters.

                #Update the finish time of the current job so that next job will know when that VM gets free.
                vm_available_time[vm["VM_ID"]] = finish_time

                #Increment VM's utilization parameter so as to not to overburden the VM beyond its capacity.
                #vm_utilization[vm["VM_ID"]] += utilization_increment

                #Now calculate the fitness value
                execution_cost = instruction_count * vm["Cost_Per_Instruction"]
                delay_penalty = max(0, (finish_time - deadline)) * delay_cost
                energy_cost = instruction_count * vm["Energy_Consumed_Per_Instruction"]

                total_cost += execution_cost
                penalty += delay_penalty
                total_energy += energy_cost

        #Return the fitness score of the chromosome
        #We convert it to negative as we want to 'minimize' the total cost required.
        #Our aim to reduce this value. 
        #So, if we get a value of say -10,000 in first generation, our goal should be to get any value > -10,000.

        #Also return the list of failed jobs for their further processing.
        return -(total_cost + penalty + total_energy), failed_jobs

    # First randomly select 'k' population. Then out of those random 'k',
    # Select the population having least negative fitness score (basically, the highest.)
    def tournament_selection(population, fitness_scores, k=3):
        selected = random.sample(list(zip(population, fitness_scores)), k)
        selected = max(selected, key=lambda x: x[1])
        return selected[0]

    # Crossover will take place randomly if the random number is less than crossover_rate given as input.
    # The point at which the crossover will happen is also random.
    def crossover(parent1, parent2, crossover_rate):
        if random.random() > crossover_rate:
            return parent1[:], parent2[:]
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    #Change the assigned VM randomly based on mutation rate.
    def mutate(chromosome, num_vms, mutation_rate=0.01):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(0, num_vms - 1)
        return chromosome

    #Initialize Population
    population = initialize_population(pop_size, num_jobs, num_vms)
    
    #Fitness scores of each population
    fitness_scores = [fitness_function(chromosome, jobs, vms)[0] for chromosome in population]

    #Loop through generations.
    for generation in range(generations):
        new_population = []
        while len(new_population) < pop_size:

            #Generate new population 
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            #Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            #Mutation
            child1 = mutate(child1, num_vms, mutation_rate)
            child2 = mutate(child2, num_vms, mutation_rate)

            #Creation of new population
            new_population.extend([child1, child2])
        population = new_population[:pop_size]

        #Calculation of fitness in current iteration
        fitness_scores = [fitness_function(chromosome, jobs, vms)[0] for chromosome in population]

        #Best Fitness of Current Generation
        best_fitness = max(fitness_scores)

        #Create a list for storing Best Fitness observed in every iteration
        best_fitness_over_time.append(best_fitness)

    #Choosing the iteration having best fitness score from last iteration. Principle of GA
    best_index = fitness_scores.index(max(fitness_scores))
    best_chromosome = population[best_index]

    #Fetching Failed Jobs from final iteration. Code needs to be updated.
    final_fitness, failed_jobs = fitness_function(best_chromosome, jobs, vms)

    vm_jobs = {vm["VM_ID"]: [] for vm in vms}
    vm_idle_times = {vm["VM_ID"]: 0 for vm in vms}
    vm_costs = {vm["VM_ID"]: 0 for vm in vms}
    vm_available_time = {vm["VM_ID"]: 0 for vm in vms}
    vm_energy = {vm["VM_ID"]: 0 for vm in vms}
    vm_waiting_times = {vm["VM_ID"]: [] for vm in vms}

    failed_job_IDS = []
    for failed_job_item in failed_jobs:
        failed_job_IDS.append(failed_job_item['Job_ID'])

    for job_index, vm_id in enumerate(best_chromosome):
        job = jobs[job_index]
        vm = vms[vm_id]

        #Excludind the Failed Jobs. Very Important.
        if job['Job_ID'] in failed_job_IDS:
            continue
    
        #Calculate the data to be included in final report.
        execution_time = job["EIC"] / vm["Million_instruction_per_second"]
        start_time = max(job["Arrival_Time"], vm_available_time[vm["VM_ID"]])
        end_time = start_time + execution_time
        waiting_time = max(0, start_time - job["Arrival_Time"])
        cost = job["EIC"] * vm["Cost_Per_Instruction"]
        energy = job["EIC"] * vm["Energy_Consumed_Per_Instruction"]

        vm_jobs[vm["VM_ID"]].append({
            "Job_ID": job["Job_ID"],
            "Start_Time": round(start_time, 2),
            "End_Time": round(end_time, 2),
            "Waiting_Time": round(waiting_time, 2),
            "Cost": round(cost, 2),
        })

        vm_costs[vm["VM_ID"]] += cost
        vm_energy[vm["VM_ID"]] += energy
        vm_waiting_times[vm["VM_ID"]].append(waiting_time)
        vm_available_time[vm["VM_ID"]] = end_time

    total_duration = max(vm_available_time.values())

    for vm_id in vm_idle_times:
        vm_idle_times[vm_id] = total_duration - sum(
            job["End_Time"] - job["Start_Time"] for job in vm_jobs[vm_id]
        )

    for vm_id in vm_waiting_times:
        vm_waiting_times[vm_id] = sum(vm_waiting_times[vm_id]) / len(vm_waiting_times[vm_id]) if vm_waiting_times[vm_id] else 0

    # Reschedule failed jobs
    rescheduled_jobs = []
    vm_utilization = {vm["VM_ID"]: 0 for vm in vms}
    for failed_job in failed_jobs:
        rescheduled = False

        execution_time = failed_job["EIC"] / vm["Million_instruction_per_second"]
        
        #Here the VMs will be checked serially. So it is possible that the starting VMs will be overloaded.
        for vm in vms:
            if (
                failed_job["Type_of_Service"] in vm["Type_of_Service"]
                and failed_job["Job_Type"] in vm["Job_Type"]
                #and vm["cpu_capacity"] < job["CPU_Requested"]
                #and vm["memory_capacity"] < job["Memory_Requested"]
                #and vm_utilization[vm_id] + utilization_increment > vm["Utilization_Capacity"]
            ):
                execution_time = failed_job["EIC"] / vm["Million_instruction_per_second"]
                start_time = max(failed_job["Arrival_Time"], vm_available_time[vm["VM_ID"]])
                end_time = start_time + execution_time
                penalty_cost = (end_time - failed_job["Deadline"]) * failed_job["Delay_Cost"]

                rescheduled_jobs.append({
                    "Job_ID": failed_job["Job_ID"],
                    "Rescheduled_VM": vm["VM_ID"],
                    "Rescheduled_Time": round(start_time, 2),
                    "Penalty": round(max(10, 10 + penalty_cost), 2), #By default for rescheduling, 5000 cost would be required.
                })

                vm_available_time[vm["VM_ID"]] = end_time
                rescheduled = True
                break

        if not rescheduled:
            rescheduled_jobs.append({
                "Job_ID": failed_job["Job_ID"],
                "Rescheduled_VM": "-- Could not find Compatible VM --",
                "Rescheduled_Time": "N/A",
                "Penalty": "N/A",
            })

    return vm_jobs, vm_idle_times, vm_costs, failed_jobs, total_duration, best_fitness_over_time, vm_energy, vm_waiting_times, rescheduled_jobs

# Parameters
pop_size = 18
generations = 300
mutation_rate = 0.03
crossover_rate = 0.9

#breakpoint()
# Run the Genetic Algorithm
vm_jobs, vm_idle_times, vm_costs, failed_jobs, total_duration, best_fitness_over_time, vm_energy, vm_waiting_times, rescheduled_jobs = genetic_algorithm(
    jobs, vms, pop_size, generations, mutation_rate, crossover_rate
)

# Save results
with open('C:\\Users\\HP\\Desktop\\output_table.txt', 'w') as file:

    #VM wise details
    for vm_id, jobs in vm_jobs.items():
        vm_id_str = f"\n Number of jobs assigned to {vm_id}: {len(jobs)}\n"
        vm_schedule_str = tabulate(jobs, headers="keys", tablefmt='grid')

        file.write(vm_id_str)
        file.write(vm_schedule_str)
        file.write('\n')

    #Failed jobs result
    cols_to_print = ["Job_ID", "Job_Type", "Deadline", "CPU_Requested", "Memory_Requested", "Failure_Reason"]
    filtered_data = [{col: row[col] for col in cols_to_print} for row in failed_jobs]
    filtered_data.sort(key=lambda x: x['Job_ID'])
    f1 = f"\n Failed Jobs Details:\n Total Jobs Failed: {len(filtered_data)}\n"
    failed_jobs_str = tabulate(filtered_data, headers="keys", tablefmt='grid')
    file.write(f1)
    file.write(failed_jobs_str)
    file.write('\n')

    rescheduled_jobs_str = tabulate(rescheduled_jobs, headers="keys", tablefmt='grid')
    file.write("\nRescheduled Jobs Details:\n")
    file.write(rescheduled_jobs_str)
    file.write('\n')

    #VM Summary
    file.write("\nVM Summary:\n")
    vm_detais_str = tabulate(
            [
                {
                    "VM": vm_id,
                    "Idle_Time": round(idle_time, 2),
                    "Cost": round(vm_costs[vm_id], 2),
                    "Energy_Spent": round(vm_energy[vm_id], 2),
                    "Avg_Waiting_Time": round(vm_waiting_times[vm_id], 2),
                }
                for vm_id, idle_time in vm_idle_times.items()
            ],
            headers="keys", tablefmt='grid'
        )
    file.write(vm_detais_str)
    file.write('\n')

    schedule_duration_str = f"\nTotal Schedule Duration: {round(total_duration, 2)} seconds"
    file.write(schedule_duration_str)
    file.write('\n')



plt.figure(figsize=(10, 6))
plt.plot([-fitness for fitness in best_fitness_over_time], marker="o", color="blue")
plt.title("Genetic Algorithm Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Score")
plt.grid(True)
plt.show()
