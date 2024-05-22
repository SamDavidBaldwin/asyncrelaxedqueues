use message_payload::MessagePayload;
use mpi::traits::*;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use crate::message_payload::VectorClock;
use crate::process_data::ProcessData;
extern crate ctrlc;
mod message_payload;
mod process_data;

fn handle_client(stream: TcpStream, tx: Sender<MessagePayload>, rank: i32) {
    let mut reader = BufReader::new(stream.try_clone().expect("Failed to clone stream"));
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                break;
            }
            Ok(_) => {
                println!("Process {} received: {}", rank, line.trim());
                // Attempt to parse the message
                if let Some(message) = parse_message(&line) {
                    println!("{:?}", message);
                    tx.send(message)
                        .expect("Failed to send parsed message to MPI thread");
                } else {
                    println!("Failed to parse message at process {}: {}", rank, line);
                }
            }
            Err(e) => {
                println!("Failed to read from client at process {}: {}", rank, e);
                break;
            }
        }
    }
}

fn start_server(port: u16, tx: Sender<MessagePayload>, rank: i32) -> std::io::Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    println!("Process {} server listening on port {}", rank, port);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let tx = tx.clone();
                thread::spawn(move || {
                    handle_client(stream, tx, rank);
                });
            }
            Err(e) => {
                println!("Failed to accept client at process {}: {}", rank, e);
            }
        }
    }
    Ok(())
}

fn parse_message(input: &str) -> Option<MessagePayload> {
    let mut process = None;
    let mut op = None;
    let mut val = None;

    // Split and parse the input
    for part in input.split(',') {
        let mut iter = part.trim().split(':');
        match (iter.next(), iter.next()) {
            (Some("process"), Some(value)) => process = value.trim().parse::<i32>().ok(),
            (Some("op"), Some(value)) => op = value.trim().parse::<i32>().ok(),
            (Some("value"), Some(value)) => val = value.trim().parse::<i32>().ok(),
            _ => {}
        }
    }

    if let (Some(process), Some(op), Some(val)) = (process, op, val) {
        Some(MessagePayload::new(
            op,
            val,
            process,
            process,
            process,
            VectorClock::default(),
        ))
    } else {
        None
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();

    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let mut process_data = ProcessData::new(rank, size);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Set up signal handler for Ctrl+C
    ctrlc::set_handler(move || {
        if rank == 0 {
            println!(
                r#"

=====================================================================

Termination request received, input Ctrl+C again to finalize shutdown...

=====================================================================

                "#
            );
        }
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let base_port = 8000; // Base port number
    let port = base_port + rank as u16; // Unique port for each process

    let (tx, rx): (Sender<MessagePayload>, Receiver<MessagePayload>) = mpsc::channel();

    // Start the server in a separate thread for each MPI process
    let tx_clone = tx.clone();
    thread::spawn(move || {
        start_server(port, tx_clone, rank).unwrap();
    });

    let mut dyn_data_buffer: Vec<MessagePayload> = Vec::new();
    dyn_data_buffer.push(MessagePayload::default());
    let mut current_index = 0;

    let mut msgs: Vec<MessagePayload> = Vec::new();

    // Predefined messages to run on startup
    // Note: Order of execution is not guaranteed
    msgs.push(MessagePayload::new(0, 69, 0, 0, 0, process_data.timestamp));

    msgs.push(MessagePayload::new(0, 420, 0, 0, 0, process_data.timestamp));

    msgs.push(MessagePayload::new(3, 0, 1, 1, 1, process_data.timestamp));

    msgs.push(MessagePayload::new(0, 70, 1, 1, 1, process_data.timestamp));

    loop {
        if current_index < dyn_data_buffer.len() {
            let recv_buf = &mut dyn_data_buffer[current_index];
            // Initiate non-blocking receives within a scope
            mpi::request::multiple_scope(1, |scope, coll| {
                let request = world.any_process().immediate_receive_into(scope, recv_buf);
                coll.add(request);

                loop {
                    // Check for ready receives
                    match coll.test_any() {
                        Some((_, status, result)) => {
                            // Handle the completion here

                            println!(
                                "Process {} received {:?} from process {}",
                                rank,
                                result,
                                status.source_rank(),
                            );

                            process_data.message_history.push(*result);
                            for msg in process_data.execute_locally(*result) {
                                msgs.push(msg);
                            }
                            break; // exit only when a receive has been processed
                        }
                        // While waiting for receives, try for external messages
                        _ => {
                            while let Ok(data) = rx.try_recv() {
                                // Echo or process the message further, here we just send to the next process in a simple ring
                                msgs.push(data);
                            }
                            // Send all avaliable messages, skipping invocations if a processes is
                            // currently Enq/Deq
                            let mut i = 0;
                            while i < msgs.len() {
                                if msgs[i].sender == rank
                                    && !(msgs[i].message == 0 && process_data.locked)
                                {
                                    if msgs[i].message == 0 || msgs[i].message == 3 {
                                        process_data.locked = true;
                                    }
                                    world.process_at_rank(msgs[i].receiver).send(&msgs[i]);

                                    // Remove the message from the list after processing
                                    msgs.remove(i);
                                } else {
                                    i += 1;
                                }
                            }
                        }
                    }
                }
            });
            dyn_data_buffer.push(MessagePayload::default());
            current_index += 1;
        }
    }
}
