#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include "utils/progress_bar.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <sys/stat.h>
#include <map>
#include <algorithm>
#include <cctype>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <output_vocab_path> <corpus_file1> [corpus_file2 ...] [--vocab-size N] [--min-freq N]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  --vocab-size N    Maximum vocabulary size (default: 10000)\n";
        std::cout << "  --min-freq N      Minimum word frequency to include (default: 2)\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " outputs/tokenizer.vocab data/pretraining/sample.txt --vocab-size 5000\n";
        return 1;
    }
    
    std::string output_path = argv[1];
    std::vector<std::string> corpus_files;
    int vocab_size = 10000;
    int min_freq = 2;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
        }
        else if (arg == "--min-freq" && i + 1 < argc) {
            min_freq = std::stoi(argv[++i]);
        }
        else {
            corpus_files.push_back(arg);
        }
    }
    
    if (corpus_files.empty()) {
        LoopOS::Utils::Logger::instance().error("BuildTokenizer", "No corpus files specified");
        return 1;
    }
    
    try {
        auto total_start = std::chrono::steady_clock::now();
        
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Building tokenizer vocabulary");
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Output path: " + output_path);
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Vocab size: " + std::to_string(vocab_size));
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Min frequency: " + std::to_string(min_freq));
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Corpus files: " + std::to_string(corpus_files.size()));
        
        std::cout << "\n";
        std::cout << "=========================================\n";
        std::cout << "  TOKENIZER TRAINING PIPELINE\n";
        std::cout << "=========================================\n\n";
        
        // Performance metrics
        std::map<std::string, double> step_times;
        
        // ===== STEP 1: Calculate total size with parallel scanning =====
        std::cout << "[1/5] Analyzing corpus files\n";
        std::cout << "--------------------------------------\n";
        auto step1_start = std::chrono::steady_clock::now();
        
        const unsigned int num_threads = std::thread::hardware_concurrency();
        std::cout << "Using " << num_threads << " threads for parallel scanning\n";
        
        std::atomic<size_t> total_size{0};
        std::atomic<size_t> files_processed{0};
        std::atomic<size_t> stat_call_count{0};
        const size_t PROGRESS_UPDATE_INTERVAL = 100;
        
        {
            LoopOS::Utils::ProgressBar size_progress(corpus_files.size(), "Scanning files", 50);
            
            // Worker function for each thread
            auto worker = [&](size_t start_idx, size_t end_idx) {
                size_t local_size = 0;
                size_t local_count = 0;
                
                for (size_t i = start_idx; i < end_idx; ++i) {
                    struct stat st;
                    if (stat(corpus_files[i].c_str(), &st) == 0) {
                        local_size += st.st_size;
                        local_count++;
                    }
                    
                    // Update progress atomically
                    size_t current = files_processed.fetch_add(1) + 1;
                    
                    // Update progress bar periodically
                    if (current % PROGRESS_UPDATE_INTERVAL == 0 || current == corpus_files.size()) {
                        size_progress.update(current);
                        
                        // Show incremental size and rate
                        double current_mb = total_size.load() / (1024.0 * 1024.0);
                        auto now = std::chrono::steady_clock::now();
                        double elapsed = std::chrono::duration<double>(now - step1_start).count();
                        double files_per_sec = elapsed > 0 ? current / elapsed : 0;
                        
                        std::cout << " | " << std::fixed << std::setprecision(1) << current_mb << " MB"
                                  << " | " << std::setprecision(0) << files_per_sec << " files/s";
                    }
                }
                
                // Add local results to global atomically
                total_size.fetch_add(local_size);
                stat_call_count.fetch_add(local_count);
            };
            
            // Split work across threads
            std::vector<std::thread> threads;
            size_t files_per_thread = corpus_files.size() / num_threads;
            
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start_idx = t * files_per_thread;
                size_t end_idx = (t == num_threads - 1) ? corpus_files.size() : (t + 1) * files_per_thread;
                threads.emplace_back(worker, start_idx, end_idx);
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            size_progress.finish();
        }
        
        auto step1_end = std::chrono::steady_clock::now();
        step_times["1_scan_files"] = std::chrono::duration<double>(step1_end - step1_start).count();
        
        double total_size_mb = total_size.load() / (1024.0 * 1024.0);
        std::cout << "\nResults:\n";
        std::cout << "  Files scanned: " << corpus_files.size() << "\n";
        std::cout << "  Total size: " << std::fixed << std::setprecision(2) << total_size_mb << " MB\n";
        std::cout << "  Time: " << std::setprecision(3) << step_times["1_scan_files"] << "s\n";
        std::cout << "  Threads used: " << num_threads << "\n";
        std::cout << "  stat() calls: " << stat_call_count.load() << "\n";
        std::cout << "  Throughput: " << std::setprecision(1) << (corpus_files.size() / step_times["1_scan_files"]) << " files/s\n\n";
        
        // Create tokenizer
        Utils::Tokenizer tokenizer;
        
        // ===== STEP 2: Process files and count word frequencies (batched parallel) =====
        std::cout << "[2/5] Reading & Tokenizing Files\n";
        std::cout << "--------------------------------------\n";
        std::cout << "Using " << num_threads << " threads for parallel processing\n";
        auto step2_start = std::chrono::steady_clock::now();
        
        std::map<std::string, int> word_freq;
        std::atomic<size_t> total_tokens{0};
        std::atomic<size_t> total_bytes_processed{0};
        std::atomic<size_t> total_lines{0};
        std::atomic<size_t> files_tokenized{0};
        std::mutex freq_mutex;
        
        const size_t BATCH_SIZE = 50;  // Process 50 files per batch
        
        {
            LoopOS::Utils::ProgressBar file_progress(corpus_files.size(), "Processing files", 50);
            
            // Worker function - processes batches of files
            auto worker = [&](size_t start_idx, size_t end_idx) {
                std::map<std::string, int> local_word_freq;
                size_t local_tokens = 0;
                size_t local_bytes = 0;
                size_t local_lines = 0;
                
                // Pre-allocate buffers for batch loading
                std::vector<std::string> file_contents;
                file_contents.reserve(BATCH_SIZE);
                
                for (size_t batch_start = start_idx; batch_start < end_idx; batch_start += BATCH_SIZE) {
                    size_t batch_end = std::min(batch_start + BATCH_SIZE, end_idx);
                    file_contents.clear();
                    
                    // PHASE 1: Batch load files into memory
                    for (size_t i = batch_start; i < batch_end; ++i) {
                        std::ifstream file(corpus_files[i], std::ios::binary | std::ios::ate);
                        if (!file.is_open()) {
                            file_contents.push_back("");
                            continue;
                        }
                        
                        // Get file size and allocate buffer
                        std::streamsize size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        
                        std::string buffer(size, '\0');
                        if (file.read(&buffer[0], size)) {
                            file_contents.push_back(std::move(buffer));
                        } else {
                            file_contents.push_back("");
                        }
                    }
                    
                    // PHASE 2: Process all loaded files in memory (much faster)
                    for (size_t i = 0; i < file_contents.size(); ++i) {
                        const std::string& content = file_contents[i];
                        if (content.empty()) continue;
                        
                        local_bytes += content.size();
                        
                        // Tokenize content
                        std::string current_word;
                        current_word.reserve(64);
                        
                        for (size_t pos = 0; pos < content.size(); ++pos) {
                            char c = content[pos];
                            
                            if (c == '\n') {
                                local_lines++;
                            }
                            
                            if (std::isspace(c) || std::ispunct(c)) {
                                if (!current_word.empty()) {
                                    // Convert to lowercase in-place
                                    for (char& ch : current_word) {
                                        ch = std::tolower(ch);
                                    }
                                    local_word_freq[current_word]++;
                                    local_tokens++;
                                    current_word.clear();
                                }
                                if (std::ispunct(c)) {
                                    local_word_freq[std::string(1, c)]++;
                                    local_tokens++;
                                }
                            } else {
                                current_word += c;
                            }
                        }
                        
                        if (!current_word.empty()) {
                            for (char& ch : current_word) {
                                ch = std::tolower(ch);
                            }
                            local_word_freq[current_word]++;
                            local_tokens++;
                        }
                    }
                    
                    // Update global progress after each batch
                    size_t current = files_tokenized.fetch_add(batch_end - batch_start) + (batch_end - batch_start);
                    total_bytes_processed.fetch_add(local_bytes);
                    
                    file_progress.update(current);
                    
                    // Show live metrics
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - step2_start).count();
                    double mb_so_far = total_bytes_processed.load() / (1024.0 * 1024.0);
                    double mb_per_sec = elapsed > 0 ? mb_so_far / elapsed : 0;
                    double files_per_sec = elapsed > 0 ? current / elapsed : 0;
                    double eta = files_per_sec > 0 ? (corpus_files.size() - current) / files_per_sec : 0;
                    
                    std::cout << " | " << std::fixed << std::setprecision(1) << mb_per_sec << " MB/s"
                              << " | " << std::setprecision(0) << files_per_sec << " files/s"
                              << " | ETA: " << std::setprecision(0) << eta << "s";
                    
                    local_bytes = 0;  // Reset for next batch
                }
                
                // Merge local results into global (protected by mutex)
                {
                    std::lock_guard<std::mutex> lock(freq_mutex);
                    for (const auto& [word, count] : local_word_freq) {
                        word_freq[word] += count;
                    }
                }
                
                total_tokens.fetch_add(local_tokens);
                total_lines.fetch_add(local_lines);
            };
            
            // Split work across threads
            std::vector<std::thread> threads;
            size_t files_per_thread = corpus_files.size() / num_threads;
            
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start_idx = t * files_per_thread;
                size_t end_idx = (t == num_threads - 1) ? corpus_files.size() : (t + 1) * files_per_thread;
                threads.emplace_back(worker, start_idx, end_idx);
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            file_progress.finish();
        }
        
        auto step2_end = std::chrono::steady_clock::now();
        step_times["2_tokenize"] = std::chrono::duration<double>(step2_end - step2_start).count();
        
        double mb_processed = total_bytes_processed.load() / (1024.0 * 1024.0);
        double tokens_per_sec = step_times["2_tokenize"] > 0 ? total_tokens.load() / step_times["2_tokenize"] : 0;
        double mb_per_sec = step_times["2_tokenize"] > 0 ? mb_processed / step_times["2_tokenize"] : 0;
        double files_per_sec = step_times["2_tokenize"] > 0 ? corpus_files.size() / step_times["2_tokenize"] : 0;
        
        std::cout << "\nResults:\n";
        std::cout << "  Total tokens: " << total_tokens.load() << "\n";
        std::cout << "  Unique words: " << word_freq.size() << "\n";
        std::cout << "  Total lines: " << total_lines.load() << "\n";
        std::cout << "  Data processed: " << std::setprecision(2) << mb_processed << " MB\n";
        std::cout << "  Time: " << std::setprecision(3) << step_times["2_tokenize"] << "s\n";
        std::cout << "Performance:\n";
        std::cout << "  Throughput: " << std::setprecision(2) << mb_per_sec << " MB/s (" << std::setprecision(1) << files_per_sec << " files/s)\n";
        std::cout << "  Token rate: " << std::setprecision(0) << tokens_per_sec << " tokens/s\n";
        std::cout << "  Threads used: " << num_threads << "\n";
        std::cout << "  Batch size: " << BATCH_SIZE << " files\n\n";
        
        // ===== STEP 3: Sort words by frequency =====
        std::cout << "[3/5] Sorting Vocabulary\n";
        std::cout << "--------------------------------------\n";
        auto step3_start = std::chrono::steady_clock::now();
        
        std::vector<std::pair<std::string, int>> sorted_words;
        sorted_words.reserve(word_freq.size());
        
        std::cout << "Building frequency list...\n";
        {
            LoopOS::Utils::ProgressBar sort_progress(word_freq.size(), "Building list", 50);
            size_t count = 0;
            for (const auto& pair : word_freq) {
                sorted_words.push_back(pair);
                if (++count % 10000 == 0 || count == word_freq.size()) {
                    sort_progress.update(count);
                }
            }
            sort_progress.finish();
        }
        
        std::cout << "\nSorting by frequency...\n";
        auto sort_start = std::chrono::steady_clock::now();
        
        // Sort with progress tracking (parallel sort for large vocabularies)
        if (sorted_words.size() > 100000) {
            // For large vocabularies, use parallel sort and track progress
            std::cout << "Using parallel sort for " << sorted_words.size() << " words...\n";
            
            // Launch sort in a separate thread so we can show progress
            std::atomic<bool> sort_done{false};
            std::thread sort_thread([&]() {
                std::sort(sorted_words.begin(), sorted_words.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
                sort_done.store(true);
            });
            
            // Show spinner while sorting
            const char spinner[] = {'|', '/', '-', '\\'};
            int spinner_idx = 0;
            while (!sort_done.load()) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - sort_start).count();
                std::cout << "\rSorting... " << spinner[spinner_idx] 
                          << " (" << std::fixed << std::setprecision(1) << elapsed << "s)";
                std::cout.flush();
                spinner_idx = (spinner_idx + 1) % 4;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            sort_thread.join();
            std::cout << "\r" << std::string(50, ' ') << "\r";  // Clear line
        } else {
            std::sort(sorted_words.begin(), sorted_words.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        
        auto sort_end = std::chrono::steady_clock::now();
        
        auto step3_end = std::chrono::steady_clock::now();
        step_times["3_sort"] = std::chrono::duration<double>(step3_end - step3_start).count();
        double sort_time = std::chrono::duration<double>(sort_end - sort_start).count();
        
        std::cout << "\nResults:\n";
        std::cout << "  Words sorted: " << sorted_words.size() << "\n";
        std::cout << "  Time: " << std::setprecision(3) << step_times["3_sort"] << "s\n";
        std::cout << "  Sort time: " << sort_time << "s\n";
        std::cout << "  Top word: '" << sorted_words[0].first << "' (freq: " << sorted_words[0].second << ")\n\n";
        
        // ===== STEP 4: Select vocabulary and build tokenizer =====
        std::cout << "[4/5] Building Vocabulary\n";
        std::cout << "--------------------------------------\n";
        auto step4_start = std::chrono::steady_clock::now();
        
        // Count how many words meet criteria
        size_t eligible_words = 0;
        int target_vocab = vocab_size - Utils::Tokenizer::FIRST_VOCAB_ID;
        for (const auto& [word, freq] : sorted_words) {
            if (freq < min_freq) break;
            if (eligible_words >= (size_t)target_vocab) break;
            eligible_words++;
        }
        
        std::cout << "Selection criteria:\n";
        std::cout << "  Min frequency: " << min_freq << "\n";
        std::cout << "  Target vocab size: " << target_vocab << "\n";
        std::cout << "  Eligible words: " << eligible_words << "\n";
        std::cout << "  Selected: " << std::min(eligible_words, (size_t)target_vocab) << " words\n\n";
        
        std::cout << "Building tokenizer structures...\n";
        tokenizer.build_vocabulary_from_files(corpus_files, vocab_size, min_freq);
        
        auto step4_end = std::chrono::steady_clock::now();
        step_times["4_build_vocab"] = std::chrono::duration<double>(step4_end - step4_start).count();
        
        std::cout << "\nResults:\n";
        std::cout << "  Final vocab size: " << tokenizer.vocab_size() << "\n";
        std::cout << "  Time: " << std::setprecision(3) << step_times["4_build_vocab"] << "s\n\n";
        
        // ===== STEP 5: Save tokenizer =====
        std::cout << "[5/5] Saving Tokenizer\n";
        std::cout << "--------------------------------------\n";
        auto step5_start = std::chrono::steady_clock::now();
        
        tokenizer.save(output_path);
        
        auto step5_end = std::chrono::steady_clock::now();
        step_times["5_save"] = std::chrono::duration<double>(step5_end - step5_start).count();
        
        std::cout << "\nResults:\n";
        std::cout << "  Saved to: " << output_path << "\n";
        std::cout << "  Time: " << std::setprecision(3) << step_times["5_save"] << "s\n\n";
        
        // ===== FINAL SUMMARY =====
        auto total_end = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        
        std::cout << "=========================================\n";
        std::cout << "  TRAINING COMPLETE\n";
        std::cout << "=========================================\n\n";
        
        std::cout << "Performance Summary:\n";
        std::cout << "--------------------\n";
        std::cout << "  [1] Scan files:      " << std::setw(8) << std::setprecision(3) 
                  << step_times["1_scan_files"] << "s  (" 
                  << std::setw(5) << std::setprecision(1) << (step_times["1_scan_files"]/total_time*100) << "%)\n";
        std::cout << "  [2] Tokenize:        " << std::setw(8) << std::setprecision(3) 
                  << step_times["2_tokenize"] << "s  (" 
                  << std::setw(5) << std::setprecision(1) << (step_times["2_tokenize"]/total_time*100) << "%)\n";
        std::cout << "  [3] Sort vocabulary: " << std::setw(8) << std::setprecision(3) 
                  << step_times["3_sort"] << "s  (" 
                  << std::setw(5) << std::setprecision(1) << (step_times["3_sort"]/total_time*100) << "%)\n";
        std::cout << "  [4] Build vocab:     " << std::setw(8) << std::setprecision(3) 
                  << step_times["4_build_vocab"] << "s  (" 
                  << std::setw(5) << std::setprecision(1) << (step_times["4_build_vocab"]/total_time*100) << "%)\n";
        std::cout << "  [5] Save:            " << std::setw(8) << std::setprecision(3) 
                  << step_times["5_save"] << "s  (" 
                  << std::setw(5) << std::setprecision(1) << (step_times["5_save"]/total_time*100) << "%)\n";
        std::cout << "  ----------------------------------------\n";
        std::cout << "  Total time:          " << std::setw(8) << std::setprecision(3) 
                  << total_time << "s\n\n";
        
        std::cout << "Output Statistics:\n";
        std::cout << "------------------\n";
        std::cout << "  Files processed: " << corpus_files.size() << "\n";
        std::cout << "  Data size: " << std::setprecision(2) << total_size_mb << " MB\n";
        std::cout << "  Tokens processed: " << total_tokens << "\n";
        std::cout << "  Unique words: " << word_freq.size() << "\n";
        std::cout << "  Final vocabulary: " << tokenizer.vocab_size() << " tokens\n";
        std::cout << "  Average throughput: " << std::setprecision(2) << (total_size_mb / total_time) << " MB/s\n\n";
        
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Tokenizer built successfully!");
        LoopOS::Utils::Logger::instance().info("BuildTokenizer", "Final vocabulary size: " + std::to_string(tokenizer.vocab_size()));
        
        // Test the tokenizer
        std::string test_text = "Hello world! This is a test.";
        auto tokens = tokenizer.encode(test_text);
        auto decoded = tokenizer.decode(tokens);
        
        std::cout << "\n=== Tokenizer Test ===\n";
        std::cout << "Original: " << test_text << "\n";
        std::cout << "Tokens: ";
        for (int token : tokens) {
            std::cout << token << " ";
        }
        std::cout << "\nDecoded: " << decoded << "\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        LoopOS::Utils::Logger::instance().error("BuildTokenizer", "Error: " + std::string(e.what()));
        return 1;
    }
}
