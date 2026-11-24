%% ========================================
% CNN dataset generation for RFUAV IQ data
% Test mode: process only pack1_0-1s.iq first.
% ========================================

clc;
clear;
close all;

%% ========== Parameters ==========
fprintf('========================================\n');
fprintf('CNN spectrogram dataset generator\n');
fprintf('========================================\n\n');

data_dir = 'C:\Users\Administrator\Desktop\数据集处理\DJI AVATA2\VTSBW=40';
fig_save_base = 'C:/Users/Administrator/Desktop/数据集处理/figure';
dataset_name = 'CNN_Dataset_test_AVATA2_VTSBW40_0_1s';

% Test first on 0-1 s.
process_only_files = {'pack4_0-1s.iq'};

fftlength = [512, 1024, 2048];
snr_gradients = [0, -5, -10, -15, -20];
overlap_ratio = 0.75;

segment_duration = 0.1;
sliding_step_ratio = 0.5;
image_size = [256, 256];
dynamic_range_db = 80;

if ~isfolder(data_dir)
    error('Data folder does not exist: %s', data_dir);
end

metadata = read_rfuav_metadata(data_dir);
fs = metadata.SampleRate;
ScaleFactor = metadata.ScaleFactor;
ref_snr_db = metadata.ReferenceSNRLevel;
drone_name = metadata.Drone;

samples_per_segment = round(fs * segment_duration);
sliding_step = round(samples_per_segment * sliding_step_ratio);

fprintf('Metadata from XML:\n');
fprintf('  Drone: %s\n', drone_name);
fprintf('  SampleRate: %.0f MHz\n', fs / 1e6);
fprintf('  CenterFrequency: %.3f GHz\n', metadata.CenterFrequency / 1e9);
fprintf('  ScaleFactor: %.6g\n', ScaleFactor);
fprintf('  ReferenceSNRLevel: %.2f dB\n\n', ref_snr_db);

fprintf('Processing config:\n');
fprintf('  Dataset: %s\n', dataset_name);
fprintf('  Files: %s\n', strjoin(process_only_files, ', '));
fprintf('  NFFT: %s\n', mat2str(fftlength));
fprintf('  SNR: %s dB\n', mat2str(snr_gradients));
fprintf('  Segment: %.0f ms, step %.0f ms\n\n', ...
        segment_duration * 1000, segment_duration * sliding_step_ratio * 1000);

dataset_dir = fullfile(fig_save_base, dataset_name);
images_dir = fullfile(dataset_dir, 'images');
if ~exist(images_dir, 'dir')
    [status, msg] = mkdir(images_dir);
    if ~status
        error('Cannot create output directory: %s\n%s', images_dir, msg);
    end
end

csv_file = fullfile(dataset_dir, 'labels.csv');
fid_csv = fopen(csv_file, 'w');
if fid_csv < 0
    error('Cannot open CSV for writing: %s', csv_file);
end
cleanup_csv = onCleanup(@() fclose(fid_csv));
fprintf(fid_csv, 'filename,drone,source_file,slice_index,slice_start_sample,snr_db,measured_snr_db,nfft\n');

%% ========== Main Loop ==========
total_images = 0;
for file_idx = 1:numel(process_only_files)
    iq_name = process_only_files{file_idx};
    file_path = fullfile(data_dir, iq_name);
    if ~isfile(file_path)
        error('IQ file does not exist: %s', file_path);
    end

    fprintf('[File %d/%d] %s\n', file_idx, numel(process_only_files), iq_name);
    dataIQ_original = read_complex_float_iq(file_path, ScaleFactor);
    fprintf('  Loaded %d complex samples (%.3f s)\n', ...
            numel(dataIQ_original), numel(dataIQ_original) / fs);

    for snr_idx = 1:numel(snr_gradients)
        target_snr = snr_gradients(snr_idx);
        [dataIQ_noisy, measured_snr_db] = add_awgn_from_reference_snr( ...
            dataIQ_original, target_snr, ref_snr_db);
        fprintf('  Target SNR %6.1f dB -> measured %.2f dB\n', ...
                target_snr, measured_snr_db);

        for nfft_idx = 1:numel(fftlength)
            nfft = fftlength(nfft_idx);
            slice_index = 0;
            start_sample = 1;

            while start_sample + samples_per_segment - 1 <= numel(dataIQ_noisy)
                end_sample = start_sample + samples_per_segment - 1;
                dataIQ_segment = dataIQ_noisy(start_sample:end_sample);

                Zxx = compute_stft_optimized(dataIQ_segment, fs, nfft, overlap_ratio);

                slice_index = slice_index + 1;
                safe_source = erase(iq_name, '.iq');
                filename = sprintf('%s_snr%+d_nfft%d_%03d.png', ...
                                   safe_source, target_snr, nfft, slice_index);
                filename = strrep(filename, '+', '');
                fig_path = fullfile(images_dir, filename);

                plot_and_save_spectrogram(Zxx, fig_path, image_size, dynamic_range_db);

                fprintf(fid_csv, '%s,%s,%s,%d,%d,%d,%.4f,%d\n', ...
                        filename, drone_name, iq_name, slice_index, ...
                        start_sample, target_snr, measured_snr_db, nfft);

                total_images = total_images + 1;
                start_sample = start_sample + sliding_step;
            end

            fprintf('    NFFT %d: %d slices\n', nfft, slice_index);
        end
    end

    clear dataIQ_original dataIQ_noisy;
end

fprintf('\n========================================\n');
fprintf('Done. Generated %d images.\n', total_images);
fprintf('Output: %s\n', dataset_dir);
fprintf('CSV: %s\n', csv_file);
fprintf('========================================\n');

%% ========== Helper Functions ==========

function metadata = read_rfuav_metadata(data_dir)
    xml_files = dir(fullfile(data_dir, '*.xml'));
    if isempty(xml_files)
        error('No XML metadata file found in %s', data_dir);
    end

    xml_path = fullfile(data_dir, xml_files(1).name);
    doc = xmlread(xml_path);

    metadata.Drone = char(get_xml_text(doc, 'Drone'));
    metadata.SampleRate = str2double(get_xml_text(doc, 'SampleRate'));
    metadata.CenterFrequency = str2double(get_xml_text(doc, 'CenterFrequency'));
    metadata.ScaleFactor = str2double(get_xml_text(doc, 'ScaleFactor'));
    metadata.ReferenceSNRLevel = str2double(get_xml_text(doc, 'ReferenceSNRLevel'));

    required = [metadata.SampleRate, metadata.CenterFrequency, ...
                metadata.ScaleFactor, metadata.ReferenceSNRLevel];
    if any(isnan(required))
        error('Invalid numeric field in XML metadata: %s', xml_path);
    end
end

function value = get_xml_text(doc, tag_name)
    nodes = doc.getElementsByTagName(tag_name);
    if nodes.getLength == 0
        error('Missing XML tag: %s', tag_name);
    end
    value = char(nodes.item(0).getTextContent);
end

function dataIQ = read_complex_float_iq(file_path, scale_factor)
    fid = fopen(file_path, 'rb');
    if fid < 0
        error('Cannot open IQ file: %s', file_path);
    end
    cleanup_fid = onCleanup(@() fclose(fid));

    data = fread(fid, inf, 'float32=>double');
    if mod(numel(data), 2) ~= 0
        warning('IQ file has an odd number of float32 values; dropping the last value.');
        data = data(1:end-1);
    end

    dataIQ = (data(1:2:end) + 1j * data(2:2:end)) ./ scale_factor;
end

function Zxx = compute_stft_optimized(dataIQ, fs, nfft, overlap_ratio)
    window_len = nfft;
    noverlap = round(window_len * overlap_ratio);
    window = hamming(window_len);

    [Zxx, ~, ~] = spectrogram(dataIQ, window, noverlap, nfft, fs);
    Zxx = fftshift(Zxx, 1);
end

function plot_and_save_spectrogram(Zxx, filepath, image_size, dynamic_range_db)
    power_dB = 10 * log10(abs(Zxx).^2 + eps);
    power_resized = imresize(power_dB, image_size, 'nearest');

    ref_db = max(power_resized(:));
    floor_db = ref_db - dynamic_range_db;
    power_resized(power_resized < floor_db) = floor_db;

    img_norm = (power_resized - floor_db) ./ dynamic_range_db;
    img_norm = min(max(img_norm, 0), 1);

    rgb = ind2rgb(round(img_norm * 255) + 1, parula(256));
    imwrite(rgb, filepath, 'png');
end

function [noisy_data, measured_snr_db] = add_awgn_from_reference_snr(dataIQ, target_snr_db, ref_snr_db)
    total_power = mean(abs(dataIQ).^2);

    ref_snr_linear = 10^(ref_snr_db / 10);
    original_noise_power = total_power / (ref_snr_linear + 1);
    signal_power = total_power - original_noise_power;

    target_snr_linear = 10^(target_snr_db / 10);
    target_noise_power = signal_power / target_snr_linear;
    additional_noise_power = max(target_noise_power - original_noise_power, 0);

    noise_std = sqrt(additional_noise_power / 2);
    noise = noise_std .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ)));
    noisy_data = dataIQ + noise;

    measured_noise_power = original_noise_power + mean(abs(noise).^2);
    measured_snr_db = 10 * log10(signal_power / measured_noise_power);
end
