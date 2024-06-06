## Player Re-Identification Pipeline
---
![[assets/player-re-id.png]]
#### Overview
- This codebase contains a data processing pipeline for spatial grounding and identification of basketball players from broadcast footage. We modify a dataset of NBA broadcast videos and labeled 2D player tracking info to containing corresponding spatial tracking information.
#### Structure and Format
- The directory structure of our final dataset is as follows:
```
root
	frames-statvu-mapped (2d-player-positions)
		NEW: mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{QUARTER}.{GAME_ID}.2D-POS.json
		OLD: mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{QUARTER}.2D-POS.json
	frames-statvu-player-tracks-mapped
		mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{QUARTER}.{GAME_ID}.2D-SPAT.json
	raw-broadcast-videos (game-replays)
		NEW: mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{QUARTER}.{GAME_ID}.RAW.json
		OLD: {GAME_ID}_mm-dd-yyyy_{AWAY_ID}_{AWAY_NAME}_{HOME_ID}_{HOME_NAME}_{PERIOD}.mp4
	hudl-game-logs
		NEW: mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{GAME_ID}.HUDL.csv
		OLD: mm-dd-yyyy.{AWAY_ABR}.{HOME_ABR}.{GAME_ID}.csv
```
- Our pipeline generates the `frames-statvu-player-tracks-mapped` directory from previously existing data located at `/mnt/sun/levlevi/nba-plus-statvu-dataset`.
- `2D-SPAT` files will have the following format:
```
{
	frame_idx [int]: {
		"alignment": {
			"confidence_score": float [0-1], 
			"predicted_time_remaining": float (seconds), 
			"predicted_quarter": int,
		"moment": {
			'moment_id': int,
			'time_remaining_in_quarter': float, 
			'time_remaining_on_shot_clock': float, 
			'player_positions': [
				{ 
					'team_id': int, 
					'player_id': int, 
					'x_position': float [0-100], 
					'y_position': float [0-50], 
					'z_position': float,
					'bounding_box': [x1, y1, x2, y2],
					'bounding_box_conf': float [0-1], 
				}
			] ... 
		} 
	}
}
```
#### Data Processing Stages
1. Unlabeled Player Tracking
```python
class BoundingBox:
	def __init__(self, id: str, x1: float, y1: float, x2: float, y2: float):
		pass

def get_player_tracks(video_path: str, out_path: str):
	"""
	Final output format:
	{
		frame_index [int] {
			[BoundingBox]
		}
	}
	"""
	return {}
```
2. Player Re-Identification
``` python
def indentify_players_from_bounding_boxes(bounding_box_data_fp: str, frames_statvu_fp: str):
	"""
	Final output format: [See `2D-SPAT` format above]
	"""
	
	return  {}
```
#### Links
- [MixSort](https://github.com/MCG-NJU/MixSort)
    - Player tracking framework built on top of [ByteTrack](https://github.com/ifzhang/ByteTrack) and [OC-SORT](https://github.com/noahcao/OC_SORT)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
    - General-purpose sports tracking repo. Pre-training data for our pipeline