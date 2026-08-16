import argparse
import re
import sys
import time

from datetime import datetime
from pathlib import Path

from gurobipy import GRB

from CE_backup import (
    Create_graph,
    Increase_CN,
    build_cycle_model,
    build_path_model,
    compute_LWCN_graph_excluding_some_edges,
    compute_LWCN_solution_excludeing_some_segments,
    compute_Length_weighted_copy_number_graph,
    compute_Length_weighted_copy_number_solution,
    convert_s_t_walk_to_segment_string,
    convert_walk_to_segment_string,
    creat_s_t_graph,
    creat_s_t_graph_Coral,
    creat_s_t_graph_for_connected_cycles,
    create_closed_walks,
    create_s_t_walks,
    match_e_constraints,
    merge_all_closed_walks,
    parse_graph_file,
    remove_s_t_from_graph,
    test_closed_walk,
    test_merged_closed_walk,
    update_the_graph,
    write_all_cycles_and_paths,
)

__version__ = "1.0.0"

TIME_LIMIT_SECONDS = 2*60*60 #2 * 60 * 60  # 2 hours
delta_F=0

def main():
    # -----------------------------------------
    # Parse command-line arguments
    # -----------------------------------------
    parser = argparse.ArgumentParser(description="Run Cycle Extractor ILP Solver.")
    parser.add_argument("--graph", required=True, help="Path to input graph file.")
    parser.add_argument("--output", required=True, help="Path to output cycles file.")
    parser.add_argument("--log_file", help="Optional, path to log file.")
    parser.add_argument("--enforce-connectivity",action="store_true",help="Enable connectivity-enforced ILP (default = off)")
    parser.add_argument("--gamma",type=float,default=0.01,help="Gamma value (default = 0.01)")
    parser.add_argument("--version",action="version",version=f"CE version {__version__}",help="Print version and exit")
    parser.add_argument("--sort-by",choices=["CopyNumber", "LWCN"],default="CopyNumber",help="How to sort cycles/paths: CopyNumber (default) or LWCN")
    parser.add_argument("--s-t-strategy",choices=["all_nodes", "intervals"],default="all_nodes",help="s/t connection strategy: 'all_nodes' (default) or 'intervals' (only interval start/end)")
    args = parser.parse_args()

    # -----------------------------------------
    # Resolve paths
    # -----------------------------------------
    graph_file_path = Path(args.graph).resolve()
    cycles_file_path = Path(args.output).resolve()
    output_dir = cycles_file_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_dir   # directory containing the graph
    ENFORCE_CONNECTIVITY = args.enforce_connectivity  # <--- HERE
    print(f"Using graph file: {graph_file_path}")
    print(f"Test/output directory: {output_dir}")

    gamma = args.gamma
    print(f"Gamma value set to: {gamma}")
    print(f"CE version: {__version__}")

    ilp_test_dir=test_dir

    amplicon_file_name = graph_file_path.name  # e.g., "amplicon1_graph.txt"
    amplicon_name = amplicon_file_name.replace("_graph.txt", "")  # e.g., "amplicon1"           


    if ENFORCE_CONNECTIVITY:
        log_file = output_dir / "ILP_Connectivity_log.txt"
    else:
        log_file = output_dir / "ILP_log.txt"
    if args.log_file:
        log_file = args.log_file

    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "w")
            self.buffer = ""

        def write(self, message):
            self.buffer += message
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                if line.strip():
                    timestamp = datetime.now().strftime("[%H:%M:%S.%f ]  ")[:-3]
                    self.log.write(f"{timestamp}{line}\n")
                    self.terminal.write(f"{line}\n")
                else:
                    self.log.write("\n")
                    self.terminal.write("\n")

        def flush(self):
            if self.buffer:
                self.log.write(self.buffer)
                self.terminal.write(self.buffer)
                self.buffer = ""
            self.log.flush()
            self.terminal.flush()
    # Open file in write mode (overwrite) or append mode ('a')
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    ilp_test_dir.mkdir(exist_ok=True)  # Create test* subdir in ILP if needed 
    print(f"CE version: {__version__}")
    print(f"Gamma value set to: {gamma}")
    print(f"Using graph file: {graph_file_path}")
    print(f"Test/output directory: {test_dir}")

    gamma = args.gamma

    if ENFORCE_CONNECTIVITY:
        print("Connectivity is enforced.")
    else:
        print("Connectivity is NOT enforced.")

    print("Creating the graph:")
    # Create the graph with integer nodes for foldbacks
    (
        G,
        nodes,
        concordant_edges,
        capacity_concordant_edges,
        discordant_edges,
        capacity_discordant_edges,
        read_count_discordant_edges,
        K_discordant_edges,
        sequence_edges,
        capacity_sequence_edges,
        length_sequence_edges,
        p_sequence_edges,
        p_concordant_edges,
        p_discordant_edges,
        path_constraints
    ) = Create_graph(graph_file_path) 

    if ENFORCE_CONNECTIVITY:
        G, nodes, discordant_edges, capacity_discordant_edges, sequence_edges, capacity_sequence_edges, K_discordant_edges, length_sequence_edges= creat_s_t_graph_for_connected_cycles(G,capacity_discordant_edges,capacity_sequence_edges,K_discordant_edges,length_sequence_edges)
    capacity_sequence_edges_original=capacity_sequence_edges.copy()
    total_solution_weight=0
    weight_graph=compute_Length_weighted_copy_number_graph(sequence_edges,capacity_sequence_edges,length_sequence_edges)
    weight_graph_excluding_some_segments=compute_LWCN_graph_excluding_some_edges(sequence_edges,capacity_sequence_edges_original,length_sequence_edges)
    p_ij = {}        
    weight_ratio_ILP_cycle=[]
    weight_ratio_ILP_cycle_excluding_some_segments=[]
    weight_ratio_ILP_path=[]
    weight_ratio_ILP_path_excluding_some_segments=[]
    Iteration=1
    Cycle_Or_Path_Number=1
    Num_paths=0
    Num_cycles=0
    All_Cycles = []
    All_Paths = []
    Disconnected_Solutions = []
    print("Start cycle extraction")
    while True: #while Iteration==1:
        ###################################################
        # ########### Solve ###############################
        ###################################################
        
        ####### Cycle solution
        ILP_Cycle_model, variables= build_cycle_model(
                        G,
                        nodes,
                        concordant_edges,
                        capacity_concordant_edges,
                        discordant_edges,
                        capacity_discordant_edges,
                        read_count_discordant_edges,
                        K_discordant_edges,
                        sequence_edges,
                        capacity_sequence_edges,
                        length_sequence_edges,
                        p_sequence_edges,
                        p_concordant_edges,
                        p_discordant_edges,
                        path_constraints,
                        ENFORCE_CONNECTIVITY,
                        gamma)
        
        ILP_Cycle_model.setParam('TimeLimit', TIME_LIMIT_SECONDS)
        print(f"Start extracting cycle {Cycle_Or_Path_Number}")
        start_time = time.time()
        ILP_Cycle_model.optimize()
        end_time = time.time()
        
        X_concordant_edge = variables["X_concordant_edge"]
        X_discordant_edge= variables["X_discordant_edge"]
        X_sequence_edge = variables["X_sequence_edge"]
        f_concordant_edge = variables["f_concordant_edge"]
        f_discordant_edge = variables["f_discordant_edge"]
        f_sequence_edge = variables["f_sequence_edge"]
        F = variables["F"]
        P = variables["P"]
        y_discordant = variables["y_discordant"]
        z_discordant = variables["z_discordant"]
        
        X_concordant_edge_val  = {k: float(v.X) for k, v in X_concordant_edge.items()}
        X_discordant_edge_val  = {k: float(v.X) for k, v in X_discordant_edge.items()}
        X_sequence_edge_val    = {k: float(v.X) for k, v in X_sequence_edge.items()}
        f_concordant_edge_val  = {k: float(v.X) for k, v in f_concordant_edge.items()}
        f_discordant_edge_val  = {k: float(v.X) for k, v in f_discordant_edge.items()}
        f_sequence_edge_val    = {k: float(v.X) for k, v in f_sequence_edge.items()}
        F_val = float(F.X)
        P_val = {k: float(v.X) for k, v in P.items()}
        
        if ILP_Cycle_model.status != GRB.OPTIMAL  or   total_solution_weight > 0.9*weight_graph or F_val<1 :
            
            Num_cycles=Cycle_Or_Path_Number-1
            break
        max_random_search_heuristic_attempts = 100

        segments, Path_Constraints = parse_graph_file(graph_file_path)
        all_closed_walks, error_in_closed_walks = create_closed_walks(sequence_edges, f_sequence_edge_val, discordant_edges, f_discordant_edge_val, 
                                concordant_edges, f_concordant_edge_val,max_random_search_heuristic_attempts,segments, F_val)
        Final_closed_walk, not_merged_walks = merge_all_closed_walks(all_closed_walks)
        Final_closed_walk_test, test_message = test_closed_walk(Final_closed_walk)
        Final_closed_walk_completion_test = test_merged_closed_walk(Final_closed_walk, sequence_edges, f_sequence_edge_val,
                discordant_edges, f_discordant_edge_val,
                concordant_edges, f_concordant_edge_val, F_val,Path=False)
        delta_F=0
        if not Final_closed_walk_completion_test:
            delta_F=Increase_CN(Final_closed_walk, F_val, capacity_sequence_edges, capacity_concordant_edges, capacity_discordant_edges)
            
            if delta_F>0:
                start_time_to_update = time.time()
                walk_edges = {
                    edge for edge_type, edge in Final_closed_walk
                    if edge_type in ("sequence edge", "concordant edge", "discordant edge")
                }
                f_sequence_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_sequence_edge_val.items()
                }
                f_concordant_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_concordant_edge_val.items()
                }
                f_discordant_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_discordant_edge_val.items()
                }
                F_val += delta_F
                end_time_to_update = time.time()
                
        final_segment_str = convert_walk_to_segment_string(Final_closed_walk, segments, copy_count=F_val)
        final_segment_str = final_segment_str.replace("Cycle=1", f"Cycle={len(all_closed_walks)+1}")
        Final_Closed_Walk_segments = final_segment_str.split("Segments=")[1].split(',')
        print(f"Finish extracting cycle {Cycle_Or_Path_Number}")
        print("Metric Calculation and Print files")
        matched_pcs = match_e_constraints(Final_Closed_Walk_segments, Path_Constraints)
        matched_index_set = set(matched_pcs)
        Solution_Type="Cycle"

        weight_ILP_cycle=compute_Length_weighted_copy_number_solution(sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk)                
        length_ILP_cycle = weight_ILP_cycle/F_val
        weight_ratio_ILP_cycle.append(weight_ILP_cycle/weight_graph)    
        weight_ILP_cycle_excluding_some_segments=compute_LWCN_solution_excludeing_some_segments(capacity_sequence_edges_original,sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk)                

        length_ILP_cycle_excluding_some_segments = weight_ILP_cycle_excluding_some_segments / F_val
        weight_ratio_ILP_cycle_excluding_some_segments.append(weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments)
        total_solution_weight += weight_ILP_cycle_excluding_some_segments
        Cycle_Info={"Cycle": Cycle_Or_Path_Number,
                    "CopyNumber": F_val,
                    "Segments": Final_Closed_Walk_segments,
                    "Length": length_ILP_cycle,
                    "Length_excluding_segments": length_ILP_cycle_excluding_some_segments,
                    "LWCN": weight_ILP_cycle,  # length-weighted copy number
                    "LWCN_excluding_segments": weight_ILP_cycle_excluding_some_segments,
                    "LWCNR": weight_ILP_cycle/weight_graph,  # length-weighted copy number
                    "LWCNR_excluding_segments": weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments,
                    "Matched_Index_Set": matched_index_set, # This is for the path constraints satisifed
                    }
        if not Final_closed_walk_completion_test :
            Disconnected_Solutions.append(Cycle_Info)
            if delta_F>0:
                 Disconnected_Solutions[-1]["Delta_F"] = delta_F
                 Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                 Disconnected_Solutions[-1]["Increase_Time"] = end_time_to_update - start_time_to_update
            else:
                 Disconnected_Solutions[-1]["Delta_F"] = 0.0
                 Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                 Disconnected_Solutions[-1]["Increase_Time"] = 0.0
        All_Cycles.append(Cycle_Info) 
        
        while not_merged_walks != []: # This while loop is for situation of disconnected cycle where they do not get merged
            Cycle_Or_Path_Number +=1
            all_closed_walks = not_merged_walks
            Final_closed_walk, not_merged_walks = merge_all_closed_walks(all_closed_walks)
            Final_closed_walk_test, test_message = test_closed_walk(Final_closed_walk)
            Final_closed_walk_completion_test = test_merged_closed_walk(Final_closed_walk, sequence_edges, f_sequence_edge_val,
                    discordant_edges, f_discordant_edge_val,
                    concordant_edges, f_concordant_edge_val, F_val, Path=False)
            F_val -= delta_F
            delta_F=0
            if not Final_closed_walk_completion_test:
                delta_F=Increase_CN(Final_closed_walk, F_val, capacity_sequence_edges, capacity_concordant_edges, capacity_discordant_edges)
                if delta_F>0:
                    start_time_to_update = time.time()
                    walk_edges = {
                        edge for edge_type, edge in Final_closed_walk
                        if edge_type in ("sequence edge", "concordant edge", "discordant edge")
                    }
                    f_sequence_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_sequence_edge_val.items()
                    }
                    f_concordant_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_concordant_edge_val.items()
                    }
                    f_discordant_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_discordant_edge_val.items()
                    }
                    F_val += delta_F
                    end_time_to_update = time.time()
            # Write Final_closed_walk as list of segments
            final_segment_str = convert_walk_to_segment_string(Final_closed_walk, segments, copy_count=F_val)
            final_segment_str = final_segment_str.replace("Cycle=1", f"Cycle={len(all_closed_walks)+1}")
            Final_Closed_Walk_segments = final_segment_str.split("Segments=")[1].split(',')
            matched_pcs = match_e_constraints(Final_Closed_Walk_segments, Path_Constraints)
            matched_index_set = set(matched_pcs)
            
            Solution_Type="Cycle"
            weight_ILP_cycle=compute_Length_weighted_copy_number_solution(sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk) 
            length_ILP_cycle = weight_ILP_cycle/F_val
            weight_ratio_ILP_cycle.append(weight_ILP_cycle/weight_graph)    
            weight_ILP_cycle_excluding_some_segments=compute_LWCN_solution_excludeing_some_segments(capacity_sequence_edges_original,sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk)                

            length_ILP_cycle_excluding_some_segments = weight_ILP_cycle_excluding_some_segments / F_val
            weight_ratio_ILP_cycle_excluding_some_segments.append(weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments)
            total_solution_weight += weight_ILP_cycle_excluding_some_segments
            Cycle_Info={"Cycle": Cycle_Or_Path_Number,
                        "CopyNumber": F_val,
                        "Segments": Final_Closed_Walk_segments,
                        "Length": length_ILP_cycle,
                        "Length_excluding_segments": length_ILP_cycle_excluding_some_segments,
                        "LWCN": weight_ILP_cycle,  # length-weighted copy number
                        "LWCN_excluding_segments": weight_ILP_cycle_excluding_some_segments,
                        "LWCNR": weight_ILP_cycle/weight_graph,  # length-weighted copy number
                        "LWCNR_excluding_segments": weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments,
                        "Matched_Index_Set": matched_index_set, # This is for the path constraints satisifed
                        }
            if not Final_closed_walk_completion_test :
                Disconnected_Solutions.append(Cycle_Info)
                if delta_F>0:
                    Disconnected_Solutions[-1]["Delta_F"] = delta_F
                    Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                    Disconnected_Solutions[-1]["Increase_Time"] = end_time_to_update - start_time_to_update
                else:
                    Disconnected_Solutions[-1]["Delta_F"] = 0.0
                    Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                    Disconnected_Solutions[-1]["Increase_Time"] = 0.0
            All_Cycles.append(Cycle_Info)
            
        
        capacity_concordant_edges, capacity_discordant_edges, capacity_sequence_edges = update_the_graph(concordant_edges, f_concordant_edge_val, X_concordant_edge_val, capacity_concordant_edges,
                                discordant_edges, f_discordant_edge_val, X_discordant_edge_val, capacity_discordant_edges,
                                sequence_edges, f_sequence_edge_val, X_sequence_edge_val, capacity_sequence_edges)

        Iteration += 1 
        Cycle_Or_Path_Number +=1

    ######## Path solution
    if ENFORCE_CONNECTIVITY:
        (
            G,
            nodes,
            discordant_edges,
            capacity_discordant_edges,
            sequence_edges,
            capacity_sequence_edges,
            K_discordant_edges,
            length_sequence_edges
        ) = remove_s_t_from_graph(
            G,
            capacity_discordant_edges,
            capacity_sequence_edges,
            K_discordant_edges,
            length_sequence_edges
        )
            

    if args.s_t_strategy == "all_nodes":
        G, nodes, discordant_edges, capacity_discordant_edges, K_discordant_edges = \
            creat_s_t_graph(G, capacity_discordant_edges, K_discordant_edges)
    else:
        sequence_edges_left = {
            edge for edge, cap in capacity_sequence_edges.items()
            if cap > 0
        }
        G, nodes, discordant_edges, capacity_discordant_edges, K_discordant_edges = \
            creat_s_t_graph_Coral(G, capacity_discordant_edges, K_discordant_edges, sequence_edges_left)

    print("Start path extraction")    
    start=time.time()
    delta_F=0
    while True: # while Iteration==1:
        ILP_Path_model, variables= build_path_model(
                        G,
                        nodes,
                        concordant_edges,
                        capacity_concordant_edges,
                        discordant_edges,
                        capacity_discordant_edges,
                        read_count_discordant_edges,
                        K_discordant_edges,
                        sequence_edges,
                        capacity_sequence_edges,
                        length_sequence_edges,
                        p_sequence_edges,
                        p_concordant_edges,
                        p_discordant_edges,
                        path_constraints,
                        ENFORCE_CONNECTIVITY,
                        gamma) 
        ILP_Path_model.setParam('TimeLimit', TIME_LIMIT_SECONDS)
        print(f"Start running Iteration {Iteration}")
        start_time = time.time()
        ILP_Path_model.optimize()
        end_time = time.time()
        print(f"Finish running Iteration {Iteration}")

        X_concordant_edge = variables["X_concordant_edge"]
        X_discordant_edge= variables["X_discordant_edge"]
        X_sequence_edge = variables["X_sequence_edge"]
        f_concordant_edge = variables["f_concordant_edge"]
        f_discordant_edge = variables["f_discordant_edge"]
        f_sequence_edge = variables["f_sequence_edge"]
        F = variables["F"]
        P = variables["P"]
        y_discordant = variables["y_discordant"]
        z_discordant = variables["z_discordant"]
        
        X_concordant_edge_val  = {k: float(v.X) for k, v in X_concordant_edge.items()}
        X_discordant_edge_val  = {k: float(v.X) for k, v in X_discordant_edge.items()}
        X_sequence_edge_val    = {k: float(v.X) for k, v in X_sequence_edge.items()}
        f_concordant_edge_val  = {k: float(v.X) for k, v in f_concordant_edge.items()}
        f_discordant_edge_val  = {k: float(v.X) for k, v in f_discordant_edge.items()}
        f_sequence_edge_val    = {k: float(v.X) for k, v in f_sequence_edge.items()}
        F_val = float(F.X)
        P_val = {k: float(v.X) for k, v in P.items()}
        
        if ILP_Path_model.status != GRB.OPTIMAL or total_solution_weight > 0.9*weight_graph or F_val<1 :
            Num_paths=Cycle_Or_Path_Number-Num_cycles-1
            break
        max_random_search_heuristic_attempts = 100       
        
        segments, Path_Constraints = parse_graph_file(graph_file_path)
        s_t_Walk, error_in_closed_walks = create_s_t_walks(sequence_edges, f_sequence_edge_val, discordant_edges, f_discordant_edge_val, 
                            concordant_edges, f_concordant_edge_val,max_random_search_heuristic_attempts,segments, F_val)

        merged_s_t_walk, not_merged_closed_walks_to_s_t_path = merge_all_closed_walks(s_t_Walk)
        final_segment_str = convert_s_t_walk_to_segment_string(s_t_Walk[0][2:-2], segments, copy_count=F_val)
        final_segment_str = re.sub(r"Path=\d+", f"Path={Cycle_Or_Path_Number}", final_segment_str)
        parts = final_segment_str.split(';', 1)  # split at first semicolon
        Final_s_t_Walk = final_segment_str.split("Segments=")[1].split(',')
        Final_s_t_Walk_segments = final_segment_str.split("Segments=")[1].split(',')
        #matched_index_set = set()
        matched_pcs = match_e_constraints(Final_s_t_Walk_segments, Path_Constraints)
        matched_index_set = set(matched_pcs)
        Final_s_t_walk_completion_test = test_merged_closed_walk(s_t_Walk[0], sequence_edges, f_sequence_edge_val,
                discordant_edges, f_discordant_edge_val,
                concordant_edges, f_concordant_edge_val, F_val,Path=True)
        F_val -= delta_F
        delta_F=0
        if not Final_s_t_walk_completion_test:
            delta_F=Increase_CN(s_t_Walk[0], F_val, capacity_sequence_edges, capacity_concordant_edges, capacity_discordant_edges)
            if delta_F>0:
                start_time_to_update = time.time()
                walk_edges = {
                    edge for edge_type, edge in Final_closed_walk
                    if edge_type in ("sequence edge", "concordant edge", "discordant edge")
                }
                f_sequence_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_sequence_edge_val.items()
                }
                f_concordant_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_concordant_edge_val.items()
                }
                f_discordant_edge_val = {
                    edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                    for edge, val in f_discordant_edge_val.items()
                }
                
                F_val += delta_F
                end_time_to_update = time.time() 
       
        Final_closed_walk_test=True
        Solution_Type="Path"
        weight_ILP_path =compute_Length_weighted_copy_number_solution(sequence_edges,f_sequence_edge_val,length_sequence_edges, s_t_Walk[0])                 
        length_ILP_path = weight_ILP_path/F_val    
        weight_ratio_ILP_path.append(weight_ILP_path/weight_graph)
        weight_ILP_path_excluding_some_segments=compute_LWCN_solution_excludeing_some_segments(capacity_sequence_edges_original,sequence_edges,f_sequence_edge_val,length_sequence_edges, s_t_Walk[0])
        length_ILP_path_excluding_some_segments = weight_ILP_path_excluding_some_segments /F_val
        weight_ratio_ILP_path_excluding_some_segments.append(weight_ILP_path_excluding_some_segments/weight_graph_excluding_some_segments)
        total_solution_weight += weight_ILP_path_excluding_some_segments

        Path_Info={"Path": Cycle_Or_Path_Number,
                "CopyNumber": F_val,
                    "Segments": Final_s_t_Walk_segments,
                    "Length": length_ILP_path,
                    "Length_excluding_segments": length_ILP_path_excluding_some_segments,
                    "LWCN": weight_ILP_path,  # length-weighted copy number
                    "LWCN_excluding_segments": weight_ILP_path_excluding_some_segments,
                    "LWCNR": weight_ILP_path/weight_graph,  # length-weighted copy number
                    "LWCNR_excluding_segments": weight_ILP_path_excluding_some_segments/weight_graph_excluding_some_segments,
                    "Matched_Index_Set": matched_index_set, # This is for the path constraints satisifed
                    }
        All_Paths.append(Path_Info) 
        
        Num_paths+=1
        Cycle_Or_Path_Number +=1
        while not_merged_closed_walks_to_s_t_path != []: # This while loop is for situation of disconnected cycle where they do not get merge to the s-t path because they are disconnected and there is not shared node or segment between the s-t path and the cycle
            #### Whatever we get in this loop is a cycle because we already got the s-t path from the solution and saved and printed it 
            Disconnected_Solutions.append(Path_Info) 
            
            all_closed_walks = not_merged_closed_walks_to_s_t_path
            Final_closed_walk, not_merged_closed_walks_to_s_t_path = merge_all_closed_walks(all_closed_walks)
            Final_closed_walk_test, test_message = test_closed_walk(Final_closed_walk)
            Final_closed_walk_completion_test = test_merged_closed_walk(Final_closed_walk, sequence_edges, f_sequence_edge_val,
                    discordant_edges, f_discordant_edge_val,
                    concordant_edges, f_concordant_edge_val, F_val, Path=False)
            F_val -= delta_F
            delta_F=0
            if not Final_closed_walk_completion_test:
                #Disconnected_Solutions.append(Cycle_Info)
                delta_F=Increase_CN(Final_closed_walk, F_val, capacity_sequence_edges, capacity_concordant_edges, capacity_discordant_edges)

                if delta_F>0:
                    start_time_to_update = time.time()
                    walk_edges = {
                        edge for edge_type, edge in Final_closed_walk
                        if edge_type in ("sequence edge", "concordant edge", "discordant edge")
                    }
                    f_sequence_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_sequence_edge_val.items()
                    }
                    f_concordant_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_concordant_edge_val.items()
                    }
                    f_discordant_edge_val = {
                        edge: val + (val / F_val) * delta_F if edge in walk_edges else val
                        for edge, val in f_discordant_edge_val.items()
                    }
                    
                    F_val += delta_F
                    end_time_to_update = time.time()
                    
            # Write Final_closed_walk as list of segments
            final_segment_str = convert_walk_to_segment_string(Final_closed_walk, segments, copy_count=F_val)
            final_segment_str = final_segment_str.replace("Cycle=1", f"Cycle={len(all_closed_walks)+1}")
            Final_Closed_Walk_segments = final_segment_str.split("Segments=")[1].split(',')
            matched_pcs = match_e_constraints(Final_Closed_Walk_segments, Path_Constraints)
            matched_index_set = set(matched_pcs)
            Solution_Type="Cycle"
            weight_ILP_cycle=compute_Length_weighted_copy_number_solution(sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk)                

            length_ILP_cycle = weight_ILP_cycle/F_val
            weight_ratio_ILP_cycle.append(weight_ILP_cycle/weight_graph)
            weight_ILP_cycle_excluding_some_segments=compute_LWCN_solution_excludeing_some_segments(capacity_sequence_edges_original,sequence_edges,f_sequence_edge_val,length_sequence_edges, Final_closed_walk)                

            length_ILP_cycle_excluding_some_segments = weight_ILP_cycle_excluding_some_segments/F_val
            weight_ratio_ILP_cycle_excluding_some_segments.append(weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments)

            total_solution_weight += weight_ILP_cycle_excluding_some_segments
           
            Cycle_Info={"Cycle": Cycle_Or_Path_Number,
                        "CopyNumber": F_val,
                        "Segments": Final_Closed_Walk_segments,
                        "Length": length_ILP_cycle,
                        "Length_excluding_segments": length_ILP_cycle_excluding_some_segments,
                        "LWCN": weight_ILP_cycle,  # length-weighted copy number
                        "LWCN_excluding_segments": weight_ILP_cycle_excluding_some_segments,
                        "LWCNR": weight_ILP_cycle/weight_graph,  # length-weighted copy number
                        "LWCNR_excluding_segments": weight_ILP_cycle_excluding_some_segments/weight_graph_excluding_some_segments,
                        "Matched_Index_Set": matched_index_set, # This is for the path constraints satisifed
                        }
            if not Final_closed_walk_completion_test :
                Disconnected_Solutions.append(Cycle_Info)
                if delta_F>0:
                    Disconnected_Solutions[-1]["Delta_F"] = delta_F
                    Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                    Disconnected_Solutions[-1]["Increase_Time"] = end_time_to_update - start_time_to_update
                else:
                    Disconnected_Solutions[-1]["Delta_F"] = 0.0
                    Disconnected_Solutions[-1]["CopyNumber_After_Increase"] = F_val
                    Disconnected_Solutions[-1]["Increase_Time"] = 0.0
            All_Cycles.append(Cycle_Info) 
            Cycle_Or_Path_Number +=1
            Num_cycles += 1
            
        capacity_concordant_edges, capacity_discordant_edges, capacity_sequence_edges = update_the_graph(concordant_edges, f_concordant_edge_val, X_concordant_edge_val, capacity_concordant_edges,
                            discordant_edges, f_discordant_edge_val, X_discordant_edge_val, capacity_discordant_edges,
                            sequence_edges, f_sequence_edge_val, X_sequence_edge_val, capacity_sequence_edges)
        Iteration += 1 

    #Determine sorting key
    if args.sort_by == "CopyNumber":
        cycle_sort_key = lambda x: x["CopyNumber"]
    else:  # LWCN
        cycle_sort_key = lambda x: x["LWCN_excluding_segments"]

    # Sort cycles and paths
    All_Cycles_Ordered = sorted(All_Cycles, key=cycle_sort_key, reverse=True)
    All_Paths_Ordered = sorted(All_Paths, key=cycle_sort_key, reverse=True)

    for i, cycle in enumerate(All_Cycles_Ordered, start=1):
        cycle["New_Order_Of_Cycle_Number"] = i
    for i, path in enumerate(All_Paths_Ordered, start=1):
        path["New_Order_Of_Path_Number"] = i+Num_cycles

    # Add new order numbers to disconnected solutions (keep the same dicts)
    for item in Disconnected_Solutions:
        if "Cycle" in item:
            # Find the cycle in All_Cycles_Ordered that matches the original number
            for cycle in All_Cycles_Ordered:
                if cycle["Cycle"] == item["Cycle"]:
                    item["New_Order_Of_Cycle_Number"] = cycle["New_Order_Of_Cycle_Number"]
                    break
        elif "Path" in item:
            for path in All_Paths_Ordered:
                if path["Path"] == item["Path"]:
                    item["New_Order_Of_Path_Number"] = path["New_Order_Of_Path_Number"]
                    break
                
    write_all_cycles_and_paths(segments, cycles_file_path, All_Cycles_Ordered, All_Paths_Ordered, Path_Constraints)

if __name__ == "__main__":
    main()
