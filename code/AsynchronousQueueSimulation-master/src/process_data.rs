use mpi::Rank;
use std::collections::VecDeque;

use crate::message_payload::{MessagePayload, VectorClock};

pub struct ProcessData {
    rank: Rank,
    world_size: i32,
    pub timestamp: VectorClock,
    pub enq_count: i32,
    pub pending_dequeues: Vec<ConfirmationList>,
    pub message_buffer: Vec<MessagePayload>,
    pub message_history: Vec<MessagePayload>,
    pub local_queue: VecDeque<(i32, Rank, VectorClock)>,
    pub locked: bool,
}

impl ProcessData {
    pub fn new(rank: Rank, size: i32) -> Self {
        ProcessData {
            rank,
            world_size: size,
            timestamp: VectorClock::new(size),
            enq_count: 0,
            pending_dequeues: Vec::new(),
            message_buffer: vec![MessagePayload::with_msg(-1); size as usize],
            message_history: Vec::new(),
            local_queue: VecDeque::new(),
            locked: false,
        }
    }

    pub fn increment_ts(&mut self) {
        self.timestamp.clock[self.rank as usize] += 1;
    }

    pub fn update_ts(&mut self, v_j: &VectorClock) {
        let max_index = self.timestamp.size;
        for i in 0..max_index {
            if v_j.clock[i] > self.timestamp.clock[i] {
                self.timestamp.clock[i] = v_j.clock[i];
            }
        }
    }

    fn find_insert_position(&self, value: &VectorClock) -> usize {
        self.local_queue
            .iter()
            .position(|x| value < &x.2)
            .unwrap_or(self.local_queue.len())
    }

    fn contains_timestamp(&self, target: &VectorClock) -> bool {
        self.pending_dequeues
            .iter()
            .any(|conf_list| conf_list.ts.clock == target.clock)
    }

    pub fn ordered_insert(&mut self, value: (i32, Rank, VectorClock)) {
        let position = self.find_insert_position(&value.2);
        self.local_queue.insert(position, value);
    }

    pub fn dequeue(&mut self, ts: VectorClock) -> Option<(i32, Rank, VectorClock)> {
        let mut oldest_index = None;
        let mut oldest_timestamp: Option<&VectorClock> = None;

        for (index, element) in self.local_queue.iter().enumerate() {
            if &element.2 < &ts {
                if oldest_timestamp.is_none() || &element.2 < oldest_timestamp.unwrap() {
                    oldest_index = Some(index);
                    oldest_timestamp = Some(&element.2);
                }
            }
        }

        if let Some(index) = oldest_index {
            self.local_queue.remove(index)
        } else {
            None
        }
    }

    pub fn insert_by_ts(&mut self, new_cl: ConfirmationList) {
        let pos = self
            .pending_dequeues
            .binary_search_by(|cl| cl.ts.compare(&new_cl.ts))
            .unwrap_or_else(|e| e);
        self.pending_dequeues.insert(pos, new_cl);
    }

    pub fn propagate_earlier_responses(&mut self) {
        for row in (1..self.pending_dequeues.len()).rev() {
            for col in 0..self.pending_dequeues[0].response_buffer.len() {
                let response = self.pending_dequeues[row].response_buffer[col];
                if response != 0 && self.pending_dequeues[row - 1].response_buffer[col] == 0 {
                    self.pending_dequeues[row - 1].response_buffer[col] = response;
                }
            }
        }
    }

    /*
    // For use in the relaxed version of this algorithm
    pub fn update_unsafes(&mut self, start_index: usize) {
        let invoker = self.pending_dequeues[start_index].invoker;
        for i in start_index..self.pending_dequeues.len() {
            self.pending_dequeues[i].response_buffer[invoker as usize] = 1;
        }
    }
    */

    pub fn execute_locally(&mut self, message_payload: MessagePayload) -> Vec<MessagePayload> {
        let mut messages_to_send: Vec<MessagePayload> = Vec::new();

        match message_payload.message {
            0 => {
                // Enq invoke
                self.enq_count = 0;
                self.increment_ts();
                for recv_rank in 0..self.world_size {
                    let message_to_send: MessagePayload = MessagePayload::new(
                        1,
                        message_payload.value,
                        message_payload.invoker,
                        self.rank,
                        recv_rank,
                        self.timestamp,
                    );
                    messages_to_send.push(message_to_send);
                }
                messages_to_send
            }
            1 => {
                // Receive EnqReq
                self.update_ts(&message_payload.time_stamp);
                self.ordered_insert((
                    message_payload.value,
                    message_payload.invoker,
                    message_payload.time_stamp,
                ));
                for confirmation_list in self.pending_dequeues.iter_mut() {
                    if confirmation_list.ts < message_payload.time_stamp {
                        confirmation_list.response_buffer[message_payload.invoker as usize] = 1;
                    }
                }
                let message_to_send: MessagePayload = MessagePayload::new(
                    2,
                    message_payload.value,
                    message_payload.invoker,
                    self.rank,
                    message_payload.invoker,
                    self.timestamp,
                );
                messages_to_send.push(message_to_send);
                messages_to_send
            }
            2 => {
                // Receive EnqAck
                self.enq_count += 1;
                if self.enq_count == self.world_size {
                    println!(
                        "Process{} done enqueueing! Current local queue: {:?}",
                        self.rank, self.local_queue
                    ); // TODO make actual return
                       // system
                    self.locked = false;
                }
                messages_to_send
            }
            3 => {
                // Deq invoke
                self.increment_ts();
                for recv_rank in 0..self.world_size {
                    let message_to_send: MessagePayload =
                        MessagePayload::new(4, 0, self.rank, self.rank, recv_rank, self.timestamp);
                    messages_to_send.push(message_to_send);
                }
                messages_to_send
            }
            4 => {
                // Receive DeqReq
                self.update_ts(&message_payload.time_stamp);
                if !self.contains_timestamp(&message_payload.time_stamp) {
                    self.insert_by_ts(ConfirmationList::new(
                        self.world_size,
                        message_payload.time_stamp,
                        message_payload.invoker,
                    ));
                }
                for recv_rank in 0..self.world_size {
                    let message_to_send: MessagePayload = MessagePayload::new(
                        5,
                        0,
                        message_payload.invoker,
                        self.rank,
                        recv_rank,
                        message_payload.time_stamp,
                    );
                    messages_to_send.push(message_to_send);
                }
                messages_to_send
            }
            5 => {
                // Receive DeqAck
                if !self.contains_timestamp(&message_payload.time_stamp) {
                    self.insert_by_ts(ConfirmationList::new(
                        self.world_size,
                        message_payload.time_stamp,
                        message_payload.invoker,
                    ));
                }
                for cl in self.pending_dequeues.iter_mut() {
                    if cl.ts == message_payload.time_stamp {
                        cl.response_buffer[message_payload.sender as usize] = 1;
                        self.propagate_earlier_responses();
                        break;
                    }
                }
                let mut i = 0;
                while i < self.pending_dequeues.len() {
                    if self.pending_dequeues[i].is_full() && !self.pending_dequeues[i].handled {
                        let ret: i32;

                        match self.dequeue(message_payload.time_stamp) {
                            Some((val, _, _)) => {
                                ret = val;
                            }
                            _ => {
                                ret = -1;
                            }
                        }
                        self.pending_dequeues[i].handled = true;
                        if self.rank == message_payload.invoker {
                            println!("Process{} dequeued {}", self.rank, ret);
                        }
                    }
                    i += 1;
                }
                self.locked = false;
                messages_to_send
            }
            _ => messages_to_send,
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct ConfirmationList {
    pub response_buffer: Vec<i32>,
    pub ts: VectorClock,
    pub invoker: Rank,
    pub handled: bool,
}

impl ConfirmationList {
    pub fn new(size: i32, deq_ts: VectorClock, deq_invoker: Rank) -> Self {
        ConfirmationList {
            response_buffer: vec![0; size as usize],
            ts: deq_ts,
            invoker: deq_invoker,
            handled: false,
        }
    }

    pub fn is_full(&self) -> bool {
        self.response_buffer.iter().all(|&x| x == 1)
    }
}
