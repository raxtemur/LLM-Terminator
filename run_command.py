#!/usr/bin/env python3
"""
Module: terminator
Description: A library for integrating an LLM with a terminal emulator.
Provides the `Terminator` class, which loads the model once and exposes a
method to run a terminal task interactively.
"""

import argparse
import json
import time
import pexpect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TerminalEmulator:
    """
    A terminal emulator that interacts with a shell using pexpect.
    """
    def __init__(self, shell='/bin/bash', prompt_regex=r'\$ '):
        """
        Initialize the terminal emulator.

        :param shell: Path to the shell (default is /bin/bash).
        :param prompt_regex: Regular expression for detecting the terminal prompt.
        """
        self.shell = shell
        self.prompt_regex = prompt_regex
        self.child = pexpect.spawn(self.shell, encoding='utf-8', echo=True)
        # Wait for the prompt to appear after starting the shell.
        self.child.expect(self.prompt_regex)

    def send_command(self, command, timeout=10):
        """
        Send a command to the terminal and return the output.

        :param command: The command to execute.
        :param timeout: Timeout in seconds to wait for the prompt.
        :return: Output of the command as a string.
        :raises TimeoutError: If waiting for the prompt times out.
        """
        self.child.sendline(command)
        try:
            self.child.expect(self.prompt_regex, timeout=timeout)
        except pexpect.TIMEOUT:
            raise TimeoutError(f"Timeout while waiting for prompt after command: {command}")
        output = self.child.before
        return output.strip()

    def get_output(self):
        """
        Retrieve asynchronous output from the terminal.

        :return: The current terminal output.
        """
        try:
            output = self.child.read_nonblocking(size=1024, timeout=1)
            return output
        except pexpect.TIMEOUT:
            return ""

    def close(self):
        """
        Close the terminal session cleanly.
        """
        self.child.sendline("exit")
        self.child.close()


class Terminator:
    """
    Terminator integrates an LLM with a terminal emulator. It loads the model
    and tokenizer once during initialization and provides a method to run tasks
    interactively.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-14B-Instruct", shell='/bin/bash',
                 prompt_regex=r'\$ ', device=None):
        """
        Initialize the Terminator instance.

        :param model_name: The name of the model to load.
        :param shell: Shell to be used by the terminal emulator.
        :param prompt_regex: Regular expression to detect the shell prompt.
        :param device: Torch device (if None, automatically set to GPU if available).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.shell = shell
        self.prompt_regex = prompt_regex
        self.terminal = TerminalEmulator(shell=self.shell, prompt_regex=self.prompt_regex)

        # System instruction guiding the LLM's behavior.
        
        self.system_instruction = (
            "You are an expert system that interacts with a terminal. Your goal is to solve the provided task. "
            "You will receive the current terminal output after each command and must generate a structured output "
            "with two fields: 'reasoning' and 'command'. "
            "The 'command' field should contain the command to be executed in the terminal. "
            "Work accurately: your terminal doesn't support interactive commands. "
            "Only one command could be executed per time. "
            "Do not include any extra text after the command. "
            "When the goal is achieved, output an empty command (i.e. \"command:\").\n\n"
            "Example of the answer:\n"
            "reasoning: I need to list all files in the current directory, usually the command \"ls\" is used for this.\n"
            "command:ls"
        )

    def _init_messages(self, task_description):
        """
        Initialize the conversation history with the system instruction and task description.

        :param task_description: The task to be solved.
        :return: List of message dictionaries.
        """
        return [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": task_description}
        ]

    def _build_input_text(self, messages):
        """
        Construct the input prompt for the LLM based on the conversation history.

        :param messages: List of message dictionaries.
        :return: A formatted string for the LLM prompt.
        """
        # Use tokenizer.apply_chat_template if available.
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "
            return prompt

    def _generate_response(self, messages):
        """
        Generate the LLM response using the current conversation history.

        :param messages: List of message dictionaries.
        :return: Generated response text.
        """
        input_text = self._build_input_text(messages)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Decode only the newly generated tokens.
        llm_output_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return llm_output_text

    def _parse_command(self, llm_output_text):
        """
        Parse the LLM output to extract the command after 'command:'.

        :param llm_output_text: The generated LLM output.
        :return: The command as a string.
        :raises ValueError: If 'command:' is not found in the output.
        """
        start = llm_output_text.find('command:')
        if start == -1:
            raise ValueError("The 'command:' field was not found in the LLM output.")
        diff = len('command:')
        command = llm_output_text[start + diff:].strip()
        return command

    def run_terminal_task(self, task_description, interactive=True):
        """
        Run the interactive terminal task. This method maintains the conversation
        with the LLM and executes commands on the terminal emulator.

        :param task_description: Description of the task to solve.
        :param interactive: If True, waits for user confirmation before each step.
        """
        messages = self._init_messages(task_description)
        print("TASK:\n", task_description, "\n")

        try:
            while True:
                # Optionally log the conversation for debugging.
                with open("log.txt", "w") as log:
                    json.dump(messages, log, indent=4)

                llm_output_text = self._generate_response(messages)
                print("\nLLM ANSWER:\n", llm_output_text)

                try:
                    command = self._parse_command(llm_output_text)
                except ValueError as e:
                    print("Error:", e)
                    break

                print("\nMODEL'S COMMAND:", command)

                if interactive:
                    user_input = input("Print y to continue (any other key to exit): ")
                    if user_input.lower() != "y":
                        break

                if not command.strip():
                    print("Task completed. Command is empty.")
                    break

                # Execute the command in the terminal emulator.
                try:
                    terminal_output = self.terminal.send_command(command)
                    print("\nTERMINAL OUTPUT:")
                    print(terminal_output)
                except TimeoutError as e:
                    terminal_output = f"Error executing command: {str(e)}. TERMINAL RESTARTED!"
                    self.terminal.close()
                    self.terminal = TerminalEmulator(shell=self.shell, prompt_regex=self.prompt_regex)
                    print(terminal_output)

                # Update conversation history.
                messages.append({"role": "assistant", "content": llm_output_text})
                messages.append({
                    "role": "user",
                    "content": f"Task reminder: {task_description}\n\nTerminal output:\n{terminal_output}"
                })
        finally:
            self.terminal.close()
            print("Session ended.")

    def close(self):
        """
        Cleanly close the terminal emulator.
        """
        self.terminal.close()


def main():
    """
    Command-line interface entry point.
    """
    parser = argparse.ArgumentParser(description="Terminator: LLM-powered Terminal Automation")
    parser.add_argument("--task_description", type=str, required=True, help="Task description")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-14B-Instruct", help="Name of the model")
    args = parser.parse_args()

    terminator = Terminator(model_name=args.model_name)
    terminator.run_terminal_task(task_description=args.task_description, interactive=True)


if __name__ == "__main__":
    main()