(* open Syntax *)
open Graph
open Utils
open Smt 

let is_blocking : Cfg.t -> Node.t -> bool
=fun cfg n-> 
  let inv = 
    match Cfg.get_invariant n cfg with
    | Some _ -> true
    | None -> false in
  if inv 
  then true
  else match Node.get_instr n with 
  (* | I_loop_entry | I_loop_exit  *)
  | I_function_exit | I_function_entry -> true 
  | _ -> false

(* 
   BubbleSort
    L2 -> L2 : 
    L2 -> L2 
*)
let dfs cfg start =
  let rec inner_dfs stack paths visited first=
    match stack with
    | [] -> paths  
    | (node, path)::tl ->
      (* let _ = match Cfg.get_invariant node cfg with
        | Some opt ->  print_endline("Invariant: "^(string_of_inv opt))
        |None -> ()
      in *)
      if is_blocking cfg node && not first 
        then inner_dfs tl (path :: paths) visited false (* blocking & not-first *)
        else
          let neighbors = NodeSet.elements (Cfg.succs node cfg) in
          let new_stack, new_visited = list_fold (fun n (stack, visited) ->
            if List.mem n visited && not (is_blocking cfg n) 
              then (stack, visited) (* Keep on [visited & non-block] *)
              else ((n, path@[n])::stack, n::visited) (* Add on [not-visited || blocking] *)
          ) neighbors (tl, visited)  in
          inner_dfs new_stack paths new_visited false
  in
  inner_dfs [(start, [start])] [] [start] true

let create_all_bp : Cfg.t -> Node.t list list
=fun cfg -> 
  let nodes = List.filter (is_blocking cfg) (Cfg.nodesof cfg) in
  list_fold (fun n res -> 
    (* print_endline ("#"^(string_of_int (List.length res))^ " ");
    let len = List.length res in
    if len > 0 then 
      List.iter (fun n -> print_string ((string_of_int (Node.get_nodeid n -23))^" ")) (List.nth res (len-1))
    else ();
    print_endline (""); *)
  res@(dfs cfg n )) nodes []
  

let create_all_vc : Node.t list list -> Fmla.t list
=fun paths -> 
  (* let inst = get_instr node *)
  ignore paths; []

let verify_partial_correctness : Syntax.pgm -> bool 
=fun pgm -> 
  let cfg = Graph.pgm2cfg pgm in 
  let basic_paths = create_all_bp cfg in
  (* print_endline ("# of paths: "^(string_of_int (List.length basic_paths)));
  List.iteri (fun i x -> 
    print_string ((string_of_int i)^". "); 
    List.iter (fun n -> print_string ((string_of_int (Node.get_nodeid n -23))^" ")) x;
    print_endline ("")
  ) basic_paths; *)
  print_endline ("############ BASIC_PATH ##########");
  List.iteri (fun i path -> 
    print_endline ((string_of_int i)^". ");
    List.iteri (fun i n-> print_endline ("\t"^(string_of_int (i+1))^". "^(Node.to_string n))) path
    ) basic_paths;
  print_endline ("##################################");
  ignore cfg; false 

(* 
  - F : Annotation
    - Function Spec : pre, post
    - R : Loop Invariant : holds at the beginning of the loop
      - R && cond : holds on entering the loop
      - R && ~cond : holds on exiting the loop
    - Assertion
  - Partial Correctness: precond && halts -> postcond
  - N VCs -> SMT solver -> true/false
  - 1 VC ~= 1 BasicPath
    - BasicPath : Cfg.t -> Node.t list list
      - Path without loop/func_call -> Path = Node.t list
      - IF : assume [prev~cond~next, prev~(~cond)~next]
      - FUNC : assume [prev~func_pre(block), 
                        prev~func_post~next] 
      - LOOP : assume [prev~loop_enter, loop_enter~next, loop_exit~next]
    - VC : Node.t list list -> smt.t
      - PRE -> wp(POST, s1; s2; .. ; sn) 
        - order: {sn, sn-1, .., s1}
        - wp (F, s) : =, assume,  / true <=> () apply PL rules 
          -  v = e : F = subst(F, v, e) {e : +, *, ...}
          -  assume c : F = c -> F
  *)

let verify_termination : Syntax.pgm -> bool 
=fun pgm -> 
  let cfg = Graph.pgm2cfg pgm in 
  ignore cfg; false 

